#!/usr/bin/python3
"""
Production Earthquake Email Notification System
Uses SeisComP native tools for data extraction and formatting
Includes travel time curves generation with ObsPy
Enhanced with mseed waveform plotting and multi-region filtering

Version: 1.1.0
Author: Mustafa Comoglu comoglu@gmail.com
- SeisComP Native + ObsPy + Multi-region
"""

import matplotlib.pyplot as plt
import os
import sys
import json
import smtplib
import subprocess
import tempfile
import signal
import socket
import math
from datetime import datetime, timezone, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.application import MIMEApplication
from pathlib import Path
import time
import logging
import configparser
import threading
from typing import List, Dict, Optional, Set, Tuple
import xml.etree.ElementTree as ET
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

try:
    from obspy import read_events, UTCDateTime, read
    from obspy.geodetics import gps2dist_azimuth, degrees2kilometers
    from obspy.taup import TauPyModel
    from obspy.clients.fdsn import Client
    OBSPY_AVAILABLE = True
except ImportError:
    OBSPY_AVAILABLE = False
    print("Warning: ObsPy not available. Travel time curves and waveform plots will be disabled.")
    print("Install with: pip install obspy")

# Version and metadata
__version__ = "1.1.0"
__author__ = "Mustafa Comoglu comoglu@gmail.com - SeisComP Native + ObsPy + Multi-region"


class ProductionLogger:
    """Production-grade logging setup"""

    @staticmethod
    def setup_logging(log_file: str = "earthquake_notifier.log",
                      log_level: str = "INFO") -> logging.Logger:
        """Setup comprehensive logging"""
        
        # Create logs directory if it doesn't exist
        log_path = Path(log_file).parent
        log_path.mkdir(parents=True, exist_ok=True)

        # Configure logging
        numeric_level = getattr(logging, log_level.upper(), logging.INFO)

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        )

        # Setup root logger
        logger = logging.getLogger('earthquake_notifier')
        logger.setLevel(numeric_level)

        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # File handler with rotation
        try:
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                log_file, maxBytes=10*1024*1024, backupCount=5
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            print(f"Could not setup file logging: {e}")

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        return logger


class GeographicFilter:
    """Geographic filtering utilities for earthquake events"""

    @staticmethod
    def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate the great circle distance between two points on Earth"""
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * \
            math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))

        # Radius of Earth in kilometers
        r = 6371

        return c * r

    @staticmethod
    def is_in_bounding_box(lat: float, lon: float,
                           min_lat: float, max_lat: float,
                           min_lon: float, max_lon: float) -> bool:
        """Check if coordinates are within bounding box"""
        return min_lat <= lat <= max_lat and min_lon <= lon <= max_lon

    @staticmethod
    def is_within_distance(lat: float, lon: float,
                           center_lat: float, center_lon: float,
                           max_distance_km: float) -> bool:
        """Check if coordinates are within distance from center point"""
        distance = GeographicFilter.haversine_distance(
            lat, lon, center_lat, center_lon)
        return distance <= max_distance_km


class MultiRegionFilter:
    """Multi-region filtering with configurable magnitude thresholds"""
    
    def __init__(self, config: configparser.ConfigParser, logger=None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.regions = self._load_regions()
    
    def _load_regions(self) -> List[Dict]:
        """Load region configurations from config file"""
        regions = []
        
        # Check for multi-region configuration
        if self.config.has_section('regions'):
            try:
                regions_json = self.config.get('regions', 'region_list', fallback='[]')
                regions = json.loads(regions_json)
                self.logger.info(f"Loaded {len(regions)} configured regions")
            except (json.JSONDecodeError, Exception) as e:
                self.logger.warning(f"Could not parse regions configuration: {e}")
                regions = []
        
        # If no multi-region config, create from legacy single-region config
        if not regions:
            filter_type = self.config.get('filtering', 'filter_type', fallback='none')
            if filter_type != 'none':
                region = {
                    'name': self.config.get('filtering', 'filter_description', fallback='Default Region'),
                    'type': filter_type,
                    'min_magnitude': float(self.config.get('service', 'min_magnitude', fallback='2.5'))
                }
                
                if filter_type == 'bounding_box':
                    region.update({
                        'min_latitude': float(self.config.get('filtering', 'min_latitude')),
                        'max_latitude': float(self.config.get('filtering', 'max_latitude')),
                        'min_longitude': float(self.config.get('filtering', 'min_longitude')),
                        'max_longitude': float(self.config.get('filtering', 'max_longitude'))
                    })
                elif filter_type == 'radial':
                    region.update({
                        'center_latitude': float(self.config.get('filtering', 'center_latitude')),
                        'center_longitude': float(self.config.get('filtering', 'center_longitude')),
                        'max_distance_km': float(self.config.get('filtering', 'max_distance_km'))
                    })
                
                regions = [region]
                self.logger.info("Using legacy single-region configuration")
        
        return regions
    
    def apply_filter(self, events: List[Dict]) -> List[Dict]:
        """Apply multi-region filtering to events"""
        if not self.regions:
            return events
        
        filtered_events = []
        
        for event in events:
            lat = event.get('Latitude')
            lon = event.get('Longitude')
            magnitude = event.get('Magnitude')
            
            if lat is None or lon is None:
                continue
            
            # Check each region
            for region in self.regions:
                if self._event_matches_region(event, region):
                    # Add region info to event
                    event['MatchedRegion'] = region['name']
                    event['RegionMinMagnitude'] = region['min_magnitude']
                    
                    # Calculate distance if radial region
                    if region['type'] == 'radial':
                        distance = GeographicFilter.haversine_distance(
                            lat, lon, 
                            region['center_latitude'], 
                            region['center_longitude']
                        )
                        event['DistanceFromCenter'] = round(distance, 1)
                    
                    filtered_events.append(event)
                    break  # Event matched, no need to check other regions
        
        if len(filtered_events) != len(events):
            self.logger.info(
                f"Multi-region filter: {len(events)} -> {len(filtered_events)} events"
            )
        
        return filtered_events
    
    def _event_matches_region(self, event: Dict, region: Dict) -> bool:
        """Check if event matches a specific region"""
        lat = event.get('Latitude')
        lon = event.get('Longitude')
        magnitude = event.get('Magnitude')
        
        # Check magnitude threshold
        if magnitude is not None and magnitude < region['min_magnitude']:
            return False
        
        # Check geographic criteria
        if region['type'] == 'bounding_box':
            return GeographicFilter.is_in_bounding_box(
                lat, lon,
                region['min_latitude'], region['max_latitude'],
                region['min_longitude'], region['max_longitude']
            )
        elif region['type'] == 'radial':
            return GeographicFilter.is_within_distance(
                lat, lon,
                region['center_latitude'], region['center_longitude'],
                region['max_distance_km']
            )
        
        return False
    
    def get_global_min_magnitude(self) -> float:
        """Get the minimum magnitude threshold across all regions"""
        if not self.regions:
            return float(self.config.get('service', 'min_magnitude', fallback='2.5'))
        
        return min(region['min_magnitude'] for region in self.regions)


class WaveformPlotter:
    """Generate waveform plots from local FDSNWS dataselect service only"""
    
    def __init__(self, config: configparser.ConfigParser, logger=None):
        # Fix: Make sure we store the config properly
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        if not OBSPY_AVAILABLE:
            self.logger.warning("ObsPy not available - waveform plots disabled")
            self.available = False
            return
        
        self.available = True
        
        # Get local FDSNWS configuration - Fix: Add fallback handling
        try:
            self.local_fdsnws_url = self.config.get('service', 'fdsnws_url', fallback='http://localhost:8080')
        except Exception as e:
            self.logger.warning(f"Could not read fdsnws_url from config: {e}")
            self.local_fdsnws_url = 'http://localhost:8080'
        
        # Parse base URL for dataselect service
        base_url = self.local_fdsnws_url.replace('http://', '').replace('https://', '')
        if base_url.endswith('/'):
            base_url = base_url[:-1]
        self.dataselect_url = f"http://{base_url}"
        
        self.logger.info(f"WaveformPlotter configured for local FDSNWS: {self.dataselect_url}")
    
    def download_waveforms(self, event_data: Dict, stations_limit: int = 10) -> Optional[str]:
        """Download waveforms from local FDSNWS dataselect service and create plot"""
        
        if not self.available:
            return None
        
        try:
            origin_time = event_data['origin_time']
            event_id = event_data['event_id']
            
            # Time window around event (with safe fallbacks)
            try:
                pre_event = int(self.config.get('advanced', 'waveform_pre_event_seconds', fallback='30'))
            except:
                pre_event = 30
            
            try:
                post_event = int(self.config.get('advanced', 'waveform_post_event_seconds', fallback='300'))
            except:
                post_event = 300
            
            starttime = origin_time - pre_event
            endtime = origin_time + post_event
            
            self.logger.info(f"Downloading waveforms for {event_id} from {starttime} to {endtime}")
            
            # Get station information from arrivals
            stations_used = []
            if 'arrivals' in event_data and event_data['arrivals']:
                # Sort by distance and take closest stations
                arrivals_by_distance = sorted(
                    event_data['arrivals'], 
                    key=lambda x: x.get('distance_deg', 999)
                )
                
                seen_stations = set()
                for arrival in arrivals_by_distance:
                    station = arrival.get('station')
                    network = arrival.get('network', '*')
                    if station and station not in seen_stations:
                        stations_used.append({
                            'network': network,
                            'station': station,
                            'distance_deg': arrival.get('distance_deg', 0),
                            'phase': arrival.get('phase', ''),
                            'arrival_time': arrival.get('time')
                        })
                        seen_stations.add(station)
                        
                        if len(stations_used) >= stations_limit:
                            break
            
            if not stations_used:
                self.logger.warning(f"No station information available for {event_id}")
                return None
            
            self.logger.info(f"Attempting to download waveforms for {len(stations_used)} stations")
            
            # Download waveforms from local FDSNWS only
            downloaded_streams = self._download_from_local_fdsnws(
                stations_used, starttime, endtime
            )
            
            if not downloaded_streams:
                self.logger.warning(f"No waveforms downloaded from local FDSNWS for event {event_id}")
                return None
            
            self.logger.info(f"Successfully downloaded {len(downloaded_streams)} waveforms for {event_id}")
            
            # Create waveform plot
            return self._create_waveform_plot(downloaded_streams, event_data)
            
        except Exception as e:
            self.logger.error(f"Error downloading waveforms: {e}")
            return None
    
    def _download_from_local_fdsnws(self, stations_used: List[Dict], 
                                   starttime, endtime) -> List:
        """Download waveforms from local FDSNWS dataselect service"""
        
        downloaded_streams = []
        
        try:
            # Create local FDSNWS client
            local_client = Client(self.dataselect_url, timeout=120)
            self.logger.info(f"Connected to local FDSNWS: {self.dataselect_url}")
            
            # Get channel preferences from config with safe fallback
            try:
                channel_preferences = self.config.get(
                    'advanced', 'waveform_channels', 
                    fallback='BHZ,HHZ,EHZ,SHZ,*Z'
                ).split(',')
            except:
                channel_preferences = ['BHZ', 'HHZ', 'EHZ', 'SHZ', '*Z']
            
            for i, station_info in enumerate(stations_used, 1):
                network = station_info['network']
                station = station_info['station']
                
                self.logger.debug(f"Downloading {i}/{len(stations_used)}: {network}.{station}")
                
                success = False
                for channel_pattern in channel_preferences:
                    channel_pattern = channel_pattern.strip()
                    
                    try:
                        self.logger.debug(f"Trying {network}.{station}.{channel_pattern}")
                        
                        st = local_client.get_waveforms(
                            network=network,
                            station=station,
                            location='*',
                            channel=channel_pattern,
                            starttime=starttime,
                            endtime=endtime
                        )
                        
                        if len(st) > 0:
                            # Take the first available trace
                            trace = st[0]
                            
                            # Add metadata
                            trace.stats.distance_deg = station_info.get('distance_deg', 0)
                            trace.stats.phase = station_info.get('phase', '')
                            trace.stats.arrival_time = station_info.get('arrival_time')
                            
                            downloaded_streams.append(trace)
                            
                            self.logger.info(
                                f"✓ Downloaded {network}.{station}.{trace.stats.channel}: "
                                f"{len(trace.data)} samples "
                                f"({trace.stats.sampling_rate} Hz)"
                            )
                            success = True
                            break
                            
                    except Exception as e:
                        self.logger.debug(f"Failed {network}.{station}.{channel_pattern}: {e}")
                        continue
                
                if not success:
                    self.logger.warning(f"✗ No waveforms found for {network}.{station}")
            
            self.logger.info(f"Local FDSNWS download completed: {len(downloaded_streams)}/{len(stations_used)} stations")
            
        except Exception as e:
            self.logger.error(f"Could not connect to local FDSNWS at {self.dataselect_url}: {e}")
            self.logger.error("Please check:")
            self.logger.error("1. SeisComP is running")
            self.logger.error("2. FDSNWS dataselect service is enabled")
            self.logger.error("3. URL is correct in config file")
        
        return downloaded_streams
    
    def _create_waveform_plot(self, traces: List, event_data: Dict) -> Optional[str]:
        """Create waveform plot from downloaded traces"""
        
        try:
            # Sort traces by distance
            traces.sort(key=lambda tr: getattr(tr.stats, 'distance_deg', 999))
            
            # Create plot
            plt.style.use('default')
            fig, axes = plt.subplots(len(traces), 1, figsize=(14, 2*len(traces)+2), sharex=True)
            
            if len(traces) == 1:
                axes = [axes]
            
            origin_time = event_data['origin_time']
            
            for i, trace in enumerate(traces):
                ax = axes[i]
                
                # Normalize and prepare data
                trace.detrend('demean')
                trace.taper(max_percentage=0.05)
                
                # Time array relative to origin time
                times = trace.times() + (trace.stats.starttime - origin_time)
                data = trace.data
                
                # Normalize amplitude
                if np.max(np.abs(data)) > 0:
                    data = data / np.max(np.abs(data))
                
                # Plot waveform
                ax.plot(times, data, 'k-', linewidth=0.8, alpha=0.8)
                ax.fill_between(times, data, alpha=0.3, color='blue')
                
                # Add station label
                distance_deg = getattr(trace.stats, 'distance_deg', 0)
                phase = getattr(trace.stats, 'phase', '')
                arrival_time = getattr(trace.stats, 'arrival_time', None)
                
                label = f"{trace.stats.network}.{trace.stats.station}.{trace.stats.channel}"
                if distance_deg:
                    label += f" ({distance_deg:.1f}°)"
                if phase:
                    label += f" [{phase}]"
                
                ax.text(0.02, 0.8, label, transform=ax.transAxes, 
                       fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
                
                # Mark theoretical P and S arrivals if we have distance
                if distance_deg and hasattr(self, 'taup_model') and self.taup_model:
                    try:
                        depth_km = event_data.get('depth_km', 10)
                        
                        # Get P arrival
                        p_arrivals = self.taup_model.get_travel_times(
                            source_depth_in_km=depth_km,
                            distance_in_degree=distance_deg,
                            phase_list=['P']
                        )
                        if p_arrivals:
                            p_time = p_arrivals[0].time
                            ax.axvline(p_time, color='red', linestyle='--', alpha=0.7, label='P')
                        
                        # Get S arrival
                        s_arrivals = self.taup_model.get_travel_times(
                            source_depth_in_km=depth_km,
                            distance_in_degree=distance_deg,
                            phase_list=['S']
                        )
                        if s_arrivals:
                            s_time = s_arrivals[0].time
                            ax.axvline(s_time, color='blue', linestyle='--', alpha=0.7, label='S')
                            
                    except Exception as e:
                        self.logger.debug(f"Could not calculate theoretical arrivals: {e}")
                
                # Mark observed arrival if available
                if arrival_time:
                    try:
                        obs_time = arrival_time - origin_time
                        ax.axvline(obs_time, color='green', linestyle='-', alpha=0.8, linewidth=2)
                    except:
                        pass
                
                ax.grid(True, alpha=0.3)
                ax.set_ylabel('Normalized\nAmplitude', fontsize=9)
                
                # Set y-limits
                ax.set_ylim(-1.2, 1.2)
            
            # Format x-axis (only on bottom plot) - with safe fallbacks
            try:
                pre_event = int(self.config.get('advanced', 'waveform_pre_event_seconds', fallback='30'))
            except:
                pre_event = 30
            
            try:
                post_event = int(self.config.get('advanced', 'waveform_post_event_seconds', fallback='300'))
            except:
                post_event = 300
            
            axes[-1].set_xlabel('Time after origin (seconds)', fontsize=12, fontweight='bold')
            axes[-1].set_xlim(-pre_event, post_event)
            
            # Add title with event information
            title_parts = []
            if event_data.get('magnitude'):
                title_parts.append(f"M{event_data['magnitude']:.1f} {event_data.get('magnitude_type', '')}")
            
            origin_time_str = event_data['origin_time'].strftime('%Y-%m-%d %H:%M:%S UTC')
            title_parts.append(origin_time_str)
            title_parts.append(f"Depth: {event_data.get('depth_km', 0):.1f}km")
            
            plt.suptitle(f"Local Waveforms: {' | '.join(title_parts)}", 
                        fontsize=14, fontweight='bold', y=0.98)
            
            # Add event info
            info_text = f"Event ID: {event_data['event_id']}\n"
            info_text += f"Location: {event_data['latitude']:.3f}°, {event_data['longitude']:.3f}°\n"
            info_text += f"Local Stations: {len(traces)}\n"
            info_text += f"Source: {self.dataselect_url}"
            
            # Add text box to first subplot
            axes[0].text(0.98, 0.95, info_text, transform=axes[0].transAxes,
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
                        verticalalignment='top', horizontalalignment='right', 
                        fontsize=9, fontfamily='monospace')
            
            plt.tight_layout()
            
            # Save plot
            safe_event_id = event_data['event_id'].replace('/', '_').replace(':', '_')
            output_file = tempfile.NamedTemporaryFile(
                suffix='.png',
                delete=False,
                prefix=f'local_waveforms_{safe_event_id}_'
            )
            output_file.close()
            
            plt.savefig(output_file.name, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close()
            
            file_size = Path(output_file.name).stat().st_size
            self.logger.info(f"Local waveforms plot saved ({file_size} bytes): {output_file.name}")
            
            return output_file.name
            
        except Exception as e:
            self.logger.error(f"Error creating waveform plot: {e}")
            plt.close()
            return None


class FDSNWSMonitor:
    """Lightweight FDSNWS event monitoring for event detection"""

    def __init__(self, base_url: str, timeout: int = 30, retries: int = 3):
        """Initialize FDSNWS monitor"""

        # Normalize URL
        url = base_url.replace(
            "fdsnws://", "").replace("http://", "").replace("https://", "")
        self.base_url = f"http://{url}".rstrip("/")
        self.event_url = f"{self.base_url}/fdsnws/event/1/query"
        self.timeout = timeout

        # Setup requests session with retries
        self.session = requests.Session()
        retry_strategy = Retry(
            total=retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        self.logger = logging.getLogger('earthquake_notifier.fdsnws')

    def get_event_list(self, config: configparser.ConfigParser, 
                      multi_region_filter: MultiRegionFilter = None, **params) -> List[Dict]:
        """Get simplified event list for monitoring new events"""

        # Use global minimum magnitude if multi-region filtering is enabled
        if multi_region_filter:
            global_min_mag = multi_region_filter.get_global_min_magnitude()
            params['minmagnitude'] = global_min_mag
            self.logger.info(f"Using global minimum magnitude: {global_min_mag}")
        else:
            # Apply legacy geographic filtering to FDSNWS query
            filter_type = config.get('filtering', 'filter_type', fallback='none').lower()

            if filter_type == 'bounding_box':
                params.update({
                    'minlatitude': config.getfloat('filtering', 'min_latitude'),
                    'maxlatitude': config.getfloat('filtering', 'max_latitude'),
                    'minlongitude': config.getfloat('filtering', 'min_longitude'),
                    'maxlongitude': config.getfloat('filtering', 'max_longitude')
                })
            elif filter_type == 'radial':
                params.update({
                    'latitude': config.getfloat('filtering', 'center_latitude'),
                    'longitude': config.getfloat('filtering', 'center_longitude'),
                    'maxradiuskm': config.getfloat('filtering', 'max_distance_km')
                })

        # Set default parameters for text format
        default_params = {
            "format": "text",
            "nodata": 404,
        }
        default_params.update(params)

        # Remove None values
        clean_params = {k: v for k, v in default_params.items()
                        if v is not None}

        try:
            self.logger.info(
                f"Fetching event list with params: {clean_params}")

            response = self.session.get(
                self.event_url,
                params=clean_params,
                timeout=self.timeout
            )
            response.raise_for_status()

            self.logger.debug(
                f"Response: {response.status_code}, Length: {len(response.text)}")

            # Parse text format response
            events = self._parse_text_format(response.text)

            # Apply multi-region filtering if configured
            if multi_region_filter:
                events = multi_region_filter.apply_filter(events)
            else:
                # Apply legacy client-side filtering if needed
                filter_type = config.get('filtering', 'filter_type', fallback='none').lower()
                if filter_type != 'none':
                    events = self._apply_client_side_filter(events, config)

            return events

        except requests.exceptions.Timeout:
            self.logger.error(f"Timeout after {self.timeout}s fetching events")
        except requests.exceptions.ConnectionError:
            self.logger.error(f"Connection error to {self.base_url}")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                self.logger.info("No events found (HTTP 404)")
            else:
                self.logger.error(f"HTTP error {e.response.status_code}: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error fetching events: {e}")

        return []

    def _parse_text_format(self, text_data: str) -> List[Dict]:
        """Parse FDSNWS text format response"""

        events = []
        lines = text_data.strip().split('\n')

        # Skip header line if present
        if lines and lines[0].startswith('#'):
            lines = lines[1:]

        for line in lines:
            if line.strip() and not line.startswith('#'):
                try:
                    parts = line.split('|')
                    if len(parts) >= 12:
                        event = {
                            'EventID': parts[0].strip(),
                            'Time': parts[1].strip(),
                            'Latitude': float(parts[2].strip()),
                            'Longitude': float(parts[3].strip()),
                            'Depth': float(parts[4].strip()) if parts[4].strip() else 0.0,
                            'Author': parts[5].strip(),
                            'Catalog': parts[6].strip(),
                            'Contributor': parts[7].strip(),
                            'ContributorID': parts[8].strip(),
                            'MagType': parts[9].strip(),
                            'Magnitude': float(parts[10].strip()) if parts[10].strip() else None,
                            'MagAuthor': parts[11].strip(),
                            'EventLocationName': parts[12].strip() if len(parts) > 12 else ''
                        }
                        events.append(event)
                except (ValueError, IndexError) as e:
                    self.logger.warning(
                        f"Error parsing line: {line[:100]}... Error: {e}")
                    continue

        self.logger.info(f"Parsed {len(events)} events from text format")
        return events

    def _apply_client_side_filter(self, events: List[Dict], config: configparser.ConfigParser) -> List[Dict]:
        """Apply client-side geographic filtering if server-side wasn't sufficient"""

        filter_type = config.get(
            'filtering', 'filter_type', fallback='none').lower()

        if filter_type == 'none':
            return events

        filtered_events = []

        for event in events:
            lat = event.get('Latitude')
            lon = event.get('Longitude')

            if lat is None or lon is None:
                continue

            include_event = False

            if filter_type == 'bounding_box':
                min_lat = config.getfloat('filtering', 'min_latitude')
                max_lat = config.getfloat('filtering', 'max_latitude')
                min_lon = config.getfloat('filtering', 'min_longitude')
                max_lon = config.getfloat('filtering', 'max_longitude')

                include_event = GeographicFilter.is_in_bounding_box(
                    lat, lon, min_lat, max_lat, min_lon, max_lon
                )

            elif filter_type == 'radial':
                center_lat = config.getfloat('filtering', 'center_latitude')
                center_lon = config.getfloat('filtering', 'center_longitude')
                max_distance = config.getfloat('filtering', 'max_distance_km')

                include_event = GeographicFilter.is_within_distance(
                    lat, lon, center_lat, center_lon, max_distance
                )

                if include_event:
                    distance = GeographicFilter.haversine_distance(
                        lat, lon, center_lat, center_lon
                    )
                    event['DistanceFromCenter'] = round(distance, 1)

            if include_event:
                filtered_events.append(event)

        if len(filtered_events) != len(events):
            self.logger.info(
                f"Client-side filter reduced events from {len(events)} to {len(filtered_events)}")

        return filtered_events


class TravelTimeCurvesGenerator:
    """Generate travel time curves visualization for earthquake events"""

    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)

        if not OBSPY_AVAILABLE:
            self.logger.warning(
                "ObsPy not available - travel time curves disabled")
            self.available = False
            return

        self.available = True

        # Initialize TauP model for theoretical travel times
        try:
            self.taup_model = TauPyModel(model="iasp91")
            self.logger.info("TauP model loaded: iasp91")
        except Exception as e:
            self.logger.warning(f"Could not load TauP model: {e}")
            self.taup_model = None

    def parse_seiscomp_xml(self, xml_file: str) -> Optional[Dict]:
        """Parse SeisComP XML file and extract event and arrival data"""
        
        if not self.available:
            return None
        
        try:
            # Fix SeisComP 0.14 and 0.13 compatibility by converting to 0.12/0.10 format
            # NOTE: KML generation happens earlier in the pipeline using original XML
            temp_xml_file = None
            try:
                with open(xml_file, 'r', encoding='utf-8') as f:
                    xml_content = f.read()
                
                conversion_needed = False
                
                # Check if this is SeisComP 0.14 format and convert to 0.12/0.12
                if 'seiscomp-schema/0.14' in xml_content:
                    self.logger.debug("Converting SeisComP 0.14 XML to 0.12/0.12 for ObsPy compatibility")
                    
                    # Replace the full namespace and version declaration
                    xml_content = xml_content.replace(
                        '<seiscomp xmlns="http://geofon.gfz.de/ns/seiscomp-schema/0.14" version="0.14">',
                        '<seiscomp xmlns="http://geofon.gfz-potsdam.de/ns/seiscomp3-schema/0.12" version="0.12">'
                    )
                    
                    # Also handle any other 0.14 schema references that might exist
                    xml_content = xml_content.replace(
                        'http://geofon.gfz.de/ns/seiscomp-schema/0.14',
                        'http://geofon.gfz-potsdam.de/ns/seiscomp3-schema/0.12'
                    )
                    conversion_needed = True
                    
                # Check if this is SeisComP 0.13 format and convert to 0.12/0.12
                elif 'seiscomp3-schema/0.13' in xml_content:
                    self.logger.debug("Converting SeisComP 0.13 XML to 0.12/0.12 for ObsPy compatibility")
                    
                    # Replace the full namespace and version declaration
                    xml_content = xml_content.replace(
                        '<seiscomp xmlns="http://geofon.gfz-potsdam.de/ns/seiscomp3-schema/0.13" version="0.13">',
                        '<seiscomp xmlns="http://geofon.gfz-potsdam.de/ns/seiscomp3-schema/0.12" version="0.12">'
                    )
                    
                    # Also handle any other 0.13 schema references that might exist
                    xml_content = xml_content.replace(
                        'http://geofon.gfz-potsdam.de/ns/seiscomp3-schema/0.13',
                        'http://geofon.gfz-potsdam.de/ns/seiscomp3-schema/0.12'
                    )
                    conversion_needed = True
                
                # If conversion was needed, create temporary file
                if conversion_needed:
                    # Create temporary file with modified content
                    import tempfile
                    temp_xml_file = tempfile.NamedTemporaryFile(
                        mode='w+', suffix='.xml', delete=False, encoding='utf-8'
                    )
                    temp_xml_file.write(xml_content)
                    temp_xml_file.close()
                    
                    # Use the temporary file for reading
                    xml_file_to_read = temp_xml_file.name
                    self.logger.debug(f"Created temporary XML file: {xml_file_to_read}")
                else:
                    xml_file_to_read = xml_file
                
                # Read events with ObsPy
                catalog = read_events(xml_file_to_read)
                
            except Exception as e:
                self.logger.warning(f"Schema conversion failed, trying direct read: {e}")
                # Fallback to direct read
                try:
                    catalog = read_events(xml_file)
                except Exception as e2:
                    self.logger.error(f"Direct XML read also failed: {e2}")
                    return None
            
            finally:
                # Clean up temporary file
                if temp_xml_file and Path(temp_xml_file.name).exists():
                    try:
                        Path(temp_xml_file.name).unlink()
                        self.logger.debug(f"Cleaned up temporary file: {temp_xml_file.name}")
                    except Exception as e:
                        self.logger.warning(f"Could not clean up temporary file: {e}")
            
            if not catalog:
                self.logger.error("No events found in XML file")
                return None
            
            event = catalog[0]
            
            # Extract basic event information
            origin = event.preferred_origin() or event.origins[0]
            magnitude = event.preferred_magnitude() or (event.magnitudes[0] if event.magnitudes else None)
            
            event_data = {
                'event_id': str(event.resource_id).split('/')[-1],
                'origin_time': origin.time,
                'latitude': origin.latitude,
                'longitude': origin.longitude,
                'depth_km': origin.depth / 1000.0 if origin.depth else 0.0,
                'magnitude': magnitude.mag if magnitude else None,
                'magnitude_type': magnitude.magnitude_type if magnitude else '',
                'arrivals': [],
                'picks': {}
            }
            
            # Create pick lookup dictionary
            for pick in event.picks:
                pick_id = str(pick.resource_id)
                event_data['picks'][pick_id] = {
                    'station': pick.waveform_id.station_code,
                    'network': pick.waveform_id.network_code,
                    'channel': pick.waveform_id.channel_code,
                    'time': pick.time,
                    'phase_hint': pick.phase_hint
                }
            
            # Extract arrival data
            for arrival in origin.arrivals:
                pick_id = str(arrival.pick_id)
                if pick_id in event_data['picks']:
                    pick_info = event_data['picks'][pick_id]
                    
                    arrival_data = {
                        'station': pick_info['station'],
                        'network': pick_info['network'],
                        'channel': pick_info['channel'],
                        'phase': arrival.phase,
                        'time': pick_info['time'],
                        'distance_deg': arrival.distance,
                        'distance_km': degrees2kilometers(arrival.distance) if arrival.distance else None,
                        'azimuth': arrival.azimuth,
                        'residual': arrival.time_residual,
                        'weight': arrival.time_weight if hasattr(arrival, 'time_weight') else 1.0
                    }
                    
                    # Calculate travel time
                    if arrival_data['time']:
                        travel_time = arrival_data['time'] - event_data['origin_time']
                        arrival_data['travel_time'] = travel_time
                    
                    event_data['arrivals'].append(arrival_data)
            
            self.logger.info(f"Parsed event {event_data['event_id']} with {len(event_data['arrivals'])} arrivals")
            return event_data
            
        except Exception as e:
            self.logger.error(f"Error parsing SeisComP XML: {e}")
            return None

    def calculate_theoretical_curves(self, depth_km: float, max_distance: float = 20.0) -> Dict:
        """Calculate theoretical P and S wave travel times"""

        if not self.taup_model:
            return {}

        try:
            distances = np.linspace(0.1, max_distance, 200)

            theoretical_curves = {
                'distances': distances,
                'phases': {}
            }

            phases = ['P', 'S', 'Pn', 'Sn', 'Pg', 'Sg']

            for phase in phases:
                times = []
                valid_distances = []

                for dist in distances:
                    try:
                        arrivals = self.taup_model.get_travel_times(
                            source_depth_in_km=depth_km,
                            distance_in_degree=dist,
                            phase_list=[phase]
                        )

                        if arrivals:
                            times.append(arrivals[0].time)
                            valid_distances.append(dist)
                        else:
                            if times:
                                times.append(np.nan)
                                valid_distances.append(dist)
                    except Exception:
                        continue

                if times:
                    theoretical_curves['phases'][phase] = {
                        'distances': np.array(valid_distances),
                        'times': np.array(times)
                    }

            self.logger.info(
                f"Calculated theoretical curves for depth {depth_km}km")
            return theoretical_curves

        except Exception as e:
            self.logger.error(f"Error calculating theoretical curves: {e}")
            return {}

    def create_travel_time_plot(self, event_data: Dict, output_file: str) -> bool:
        """Create travel time curves plot"""

        try:
            plt.style.use('default')
            fig, ax = plt.subplots(figsize=(12, 8))

            arrivals = event_data['arrivals']
            if not arrivals:
                self.logger.warning("No arrivals to plot")
                return False

            # Separate P and S phases
            p_phases = []
            s_phases = []
            other_phases = []

            for arrival in arrivals:
                if arrival['distance_deg'] and arrival['travel_time']:
                    phase = arrival['phase'].upper()
                    travel_time_sec = arrival['travel_time']

                    if 'P' in phase and 'S' not in phase:
                        p_phases.append(
                            (arrival['distance_deg'], travel_time_sec, arrival))
                    elif 'S' in phase:
                        s_phases.append(
                            (arrival['distance_deg'], travel_time_sec, arrival))
                    else:
                        other_phases.append(
                            (arrival['distance_deg'], travel_time_sec, arrival))

            # Plot observed arrivals
            if p_phases:
                p_dist, p_times, p_arrivals = zip(*p_phases)
                ax.scatter(p_dist, p_times, c='red', marker='o', s=60, 
                        label=f'P phases ({len(p_phases)})', alpha=0.8, edgecolors='darkred')
                
                # Add station names for P arrivals
                for i, (dist, time, arrival) in enumerate(p_phases):
                    station_name = arrival['station']
                    ax.annotate(station_name, 
                            (dist, time), 
                            xytext=(5, 5), 
                            textcoords='offset points',
                            fontsize=8,
                            alpha=0.8,
                            color='darkred',
                            weight='bold')

            if s_phases:
                s_dist, s_times, s_arrivals = zip(*s_phases)
                ax.scatter(s_dist, s_times, c='blue', marker='s', s=60, 
                        label=f'S phases ({len(s_phases)})', alpha=0.8, edgecolors='darkblue')
                
                # Add station names for S arrivals
                for i, (dist, time, arrival) in enumerate(s_phases):
                    station_name = arrival['station']
                    ax.annotate(station_name, 
                            (dist, time), 
                            xytext=(-8, -12), 
                            textcoords='offset points',
                            fontsize=8,
                            alpha=0.8,
                            color='darkblue',
                            weight='bold')

            if other_phases:
                o_dist, o_times, o_arrivals = zip(*other_phases)
                ax.scatter(o_dist, o_times, c='green', marker='^', s=60, 
                        label=f'Other phases ({len(other_phases)})', alpha=0.8, edgecolors='darkgreen')
                
                # Add station names for other phases
                for i, (dist, time, arrival) in enumerate(other_phases):
                    station_name = arrival['station']
                    phase_name = arrival['phase']
                    # Include phase name for other phases to distinguish them
                    label_text = f"{station_name}({phase_name})"
                    ax.annotate(label_text, 
                            (dist, time), 
                            xytext=(8, -8), 
                            textcoords='offset points',
                            fontsize=7,
                            alpha=0.8,
                            color='darkgreen',
                            weight='normal')

            # Plot theoretical curves if available
            max_distance = max([a['distance_deg']
                               for a in arrivals if a['distance_deg']]) * 1.1
            theoretical = self.calculate_theoretical_curves(
                event_data['depth_km'], max_distance)

            if theoretical:
                phase_colors = {
                    'P': 'red', 'Pn': 'darkred', 'Pg': 'lightcoral',
                    'S': 'blue', 'Sn': 'darkblue', 'Sg': 'lightblue'
                }

                for phase_name, phase_data in theoretical['phases'].items():
                    if len(phase_data['times']) > 0:
                        color = phase_colors.get(phase_name, 'gray')
                        linestyle = '-' if phase_name in ['P', 'S'] else '--'
                        alpha = 0.8 if phase_name in ['P', 'S'] else 0.5

                        valid_mask = ~np.isnan(phase_data['times'])
                        if np.any(valid_mask):
                            ax.plot(phase_data['distances'][valid_mask],
                                    phase_data['times'][valid_mask],
                                    color=color, linestyle=linestyle, alpha=alpha,
                                    linewidth=2 if phase_name in [
                                        'P', 'S'] else 1,
                                    label=f'{phase_name} theoretical')

            # Formatting and labels
            ax.set_xlabel('Distance (degrees)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Travel Time (seconds)',
                          fontsize=12, fontweight='bold')

            # Title with event information
            title_parts = []
            title_parts.append(f"Travel Time Curves")

            if event_data['magnitude']:
                title_parts.append(
                    f"M{event_data['magnitude']:.1f} {event_data['magnitude_type']}")

            origin_time_str = event_data['origin_time'].strftime(
                '%Y-%m-%d %H:%M:%S UTC')
            title_parts.append(f"{origin_time_str}")

            title_parts.append(f"Depth: {event_data['depth_km']:.1f}km")

            plt.title(' | '.join(title_parts), fontsize=14,
                      fontweight='bold', pad=20)

            # Add event info text box
            info_text = f"Event ID: {event_data['event_id']}\n"
            info_text += f"Location: {event_data['latitude']:.3f}°, {event_data['longitude']:.3f}°\n"
            info_text += f"Total Stations: {len(set(a['station'] for a in arrivals))}\n"
            info_text += f"Total Phases: {len(arrivals)}"

            ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.5",
                              facecolor="lightgray", alpha=0.8),
                    verticalalignment='top', fontsize=10, fontfamily='monospace')

            # Legend
            ax.legend(loc='lower right', frameon=True,
                      fancybox=True, shadow=True)

            # Grid and styling
            ax.grid(True, alpha=0.3)
            ax.set_axisbelow(True)

            # Set reasonable axis limits
            if arrivals:
                max_time = max([a['travel_time']
                               for a in arrivals if a['travel_time']]) * 1.1
                ax.set_ylim(0, max_time)
                ax.set_xlim(0, max_distance)

            plt.tight_layout()

            # Save the plot
            plt.savefig(output_file, dpi=300, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            plt.close()

            self.logger.info(f"Travel time curves plot saved to {output_file}")
            return True

        except Exception as e:
            self.logger.error(f"Error creating travel time plot: {e}")
            plt.close()
            return False

    def generate_email_attachment(self, xml_file: str, event_id: str = None) -> Optional[str]:
        """Generate travel time curves plot as email attachment"""

        if not self.available:
            return None

        try:
            event_data = self.parse_seiscomp_xml(xml_file)
            if not event_data:
                return None

            if event_id:
                safe_event_id = event_id.replace('/', '_').replace(':', '_')
            else:
                safe_event_id = event_data['event_id'].replace(
                    '/', '_').replace(':', '_')

            output_file = tempfile.NamedTemporaryFile(
                suffix='.png',
                delete=False,
                prefix=f'travel_times_{safe_event_id}_'
            )
            output_file.close()

            success = self.create_travel_time_plot(
                event_data, output_file.name)

            if success:
                file_size = Path(output_file.name).stat().st_size
                self.logger.info(
                    f"Travel time curves generated ({file_size} bytes): {output_file.name}")
                return output_file.name
            else:
                try:
                    Path(output_file.name).unlink()
                except:
                    pass
                return None

        except Exception as e:
            self.logger.error(f"Error generating travel time curves: {e}")
            return None


class SeisCompToolchain:
    """SeisComP native toolchain for event processing"""

    def __init__(self, config: configparser.ConfigParser):
            self.config = config
            self.logger = logging.getLogger('earthquake_notifier.seiscomp')

            # Get SeisComP paths
            self.seiscomp_root = self.config.get(
                'seiscomp', 'seiscomp_root', fallback='/opt/seiscomp')
            self.scxmldump_path = self.config.get('seiscomp', 'scxmldump_path',
                                                fallback=f'{self.seiscomp_root}/bin/scxmldump')
            self.scmapcut_path = self.config.get('seiscomp', 'scmapcut_path',
                                                fallback=f'{self.seiscomp_root}/bin/scmapcut')
            self.scbulletin_path = self.config.get('seiscomp', 'scbulletin_path',
                                                fallback=f'{self.seiscomp_root}/bin/scbulletin')

            # Database connection
            self.database_url = self.config.get('seiscomp', 'database_url',
                                                fallback='mysql://sysop:sysop@localhost/seiscomp')

            # Initialize travel time curves generator
            self.curves_generator = TravelTimeCurvesGenerator(self.logger)
            
            # Initialize waveform plotter - FIXED: Pass config properly
            self.waveform_plotter = WaveformPlotter(self.config, self.logger)

            # Verify tools exist
            self._verify_tools()

    def _verify_tools(self):
        """Verify SeisComP tools are available"""

        tools = {
            'scxmldump': self.scxmldump_path,
            'scmapcut': self.scmapcut_path,
            'scbulletin': self.scbulletin_path
        }

        for tool, path in tools.items():
            if Path(path).exists():
                self.logger.info(f"✓ {tool} available at {path}")
            else:
                self.logger.error(f"✗ {tool} not found at {path}")
                raise FileNotFoundError(
                    f"SeisComP tool not found: {tool} at {path}")

        # Check ObsPy availability for travel time curves and waveforms
        if OBSPY_AVAILABLE and self.curves_generator.available:
            self.logger.info("✓ ObsPy available - travel time curves enabled")
        else:
            self.logger.warning(
                "✗ ObsPy not available - travel time curves disabled")
            
        if OBSPY_AVAILABLE and self.waveform_plotter.available:
            self.logger.info("✓ ObsPy available - waveform plots enabled")
        else:
            self.logger.warning(
                "✗ ObsPy not available - waveform plots disabled")

    def extract_event_xml(self, event_id: str) -> Optional[str]:
        """Extract complete event XML using scxmldump"""

        try:
            xml_file = tempfile.NamedTemporaryFile(
                mode='w+', suffix='.xml', delete=False,
                prefix=f'event_{event_id.replace("/", "_")}_'
            )
            xml_file.close()

            cmd = [
                self.scxmldump_path,
                '-PAMfp',
                '-d', self.database_url,
                '-E', event_id,
                '-o', xml_file.name
            ]

            self.logger.info(f"Extracting event XML: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.returncode == 0:
                if Path(xml_file.name).exists() and Path(xml_file.name).stat().st_size > 0:
                    file_size = Path(xml_file.name).stat().st_size
                    self.logger.info(
                        f"Event XML extracted successfully ({file_size} bytes): {xml_file.name}")
                    return xml_file.name
                else:
                    self.logger.error(
                        f"scxmldump produced empty file for event {event_id}")
            else:
                self.logger.error(
                    f"scxmldump failed (code {result.returncode}): {result.stderr}")
                if result.stdout:
                    self.logger.debug(f"scxmldump stdout: {result.stdout}")

            try:
                Path(xml_file.name).unlink()
            except:
                pass

            return None

        except subprocess.TimeoutExpired:
            self.logger.error(f"scxmldump timeout for event {event_id}")
        except Exception as e:
            self.logger.error(
                f"Error extracting event XML for {event_id}: {e}")

        return None

    def generate_map(self, xml_file: str, event_id: str) -> Optional[str]:
        """Generate earthquake map using scmapcut with event XML"""

        try:
            map_file = tempfile.NamedTemporaryFile(
                suffix='.png', delete=False,
                prefix=f'map_{event_id.replace("/", "_")}_'
            )
            map_file.close()

            cmd = [
                self.scmapcut_path,
                '-E', event_id,
                '--ep', xml_file,
                '-m', '3.0',
                '-d', '1024x768',
                '-o', map_file.name
            ]

            self.logger.info(f"Generating map: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.returncode == 0:
                if Path(map_file.name).exists() and Path(map_file.name).stat().st_size > 0:
                    file_size = Path(map_file.name).stat().st_size
                    self.logger.info(
                        f"Map generated successfully ({file_size} bytes): {map_file.name}")
                    return map_file.name
                else:
                    self.logger.error(
                        f"scmapcut produced empty file for event {event_id}")
            else:
                self.logger.error(
                    f"scmapcut failed (code {result.returncode}): {result.stderr}")
                if result.stdout:
                    self.logger.debug(f"scmapcut stdout: {result.stdout}")

            try:
                Path(map_file.name).unlink()
            except:
                pass

            return None

        except subprocess.TimeoutExpired:
            self.logger.error(f"scmapcut timeout for event {event_id}")
        except Exception as e:
            self.logger.error(f"Error generating map for {event_id}: {e}")

        return None

    def generate_bulletin(self, xml_file: str, event_id: str) -> Optional[str]:
        """Generate earthquake bulletin using scbulletin"""

        try:
            cmd = [
                self.scbulletin_path,
                '-i', xml_file,
                '-3',
                '-k',
                '-e'
            ]

            self.logger.info(f"Generating bulletin: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0 and result.stdout:
                self.logger.info(
                    f"Bulletin generated successfully for event {event_id}")
                return result.stdout
            else:
                self.logger.error(
                    f"scbulletin failed (code {result.returncode}): {result.stderr}")
                if result.stdout:
                    self.logger.debug(f"scbulletin stdout: {result.stdout}")

            return None

        except subprocess.TimeoutExpired:
            self.logger.error(f"scbulletin timeout for event {event_id}")
        except Exception as e:
            self.logger.error(f"Error generating bulletin for {event_id}: {e}")

        return None

    def generate_travel_time_curves(self, xml_file: str, event_id: str) -> Optional[str]:
        """Generate travel time curves plot using ObsPy"""

        try:
            return self.curves_generator.generate_email_attachment(xml_file, event_id)
        except Exception as e:
            self.logger.error(
                f"Error generating travel time curves for {event_id}: {e}")
            return None

    def generate_waveform_plots(self, xml_file: str, event_id: str) -> Optional[str]:
        """Generate waveform plots using ObsPy"""
        
        try:
            if not self.waveform_plotter.available:
                return None
                
            # Parse event data from XML to get station information
            event_data = self.curves_generator.parse_seiscomp_xml(xml_file)
            if not event_data:
                self.logger.warning(f"Could not parse event data for waveform plots: {event_id}")
                return None
                
            # Set up TauP model for waveform plotter if available
            if hasattr(self.curves_generator, 'taup_model') and self.curves_generator.taup_model:
                self.waveform_plotter.taup_model = self.curves_generator.taup_model
                
            return self.waveform_plotter.download_waveforms(event_data)
            
        except Exception as e:
            self.logger.error(f"Error generating waveform plots for {event_id}: {e}")
            return None

    def generate_kml(self, xml_file: str, event_id: str) -> Optional[str]:
        """Generate KML file using scbulletin"""
        
        try:
            kml_file = tempfile.NamedTemporaryFile(
                suffix='.kml', delete=False, 
                prefix=f'event_{event_id.replace("/", "_")}_'
            )
            kml_file.close()
            
            cmd = [
                self.scbulletin_path,
                '-i', xml_file,
                '--kml',
                '-o', kml_file.name
            ]
            
            self.logger.info(f"Generating KML: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=60
            )
            
            if result.returncode == 0:
                if Path(kml_file.name).exists() and Path(kml_file.name).stat().st_size > 0:
                    file_size = Path(kml_file.name).stat().st_size
                    self.logger.info(f"KML generated successfully ({file_size} bytes): {kml_file.name}")
                    return kml_file.name
                else:
                    self.logger.error(f"scbulletin produced empty KML file for event {event_id}")
            else:
                self.logger.error(f"scbulletin KML failed (code {result.returncode}): {result.stderr}")
                if result.stdout:
                    self.logger.debug(f"scbulletin KML stdout: {result.stdout}")
            
            try:
                Path(kml_file.name).unlink()
            except:
                pass
            
            return None
            
        except subprocess.TimeoutExpired:
            self.logger.error(f"scbulletin KML timeout for event {event_id}")
        except Exception as e:
            self.logger.error(f"Error generating KML for {event_id}: {e}")
        
        return None

    def cleanup_temp_files(self, *files):
        """Clean up temporary files"""
        for file_path in files:
            if file_path and Path(file_path).exists():
                try:
                    Path(file_path).unlink()
                    self.logger.debug(f"Cleaned up: {file_path}")
                except Exception as e:
                    self.logger.warning(f"Could not clean up {file_path}: {e}")


class EmailNotifier:
    """Production email notification system"""

    def __init__(self, config: configparser.ConfigParser):
        self.config = config
        self.logger = logging.getLogger('earthquake_notifier.email')

        self.test_mode = self._is_test_mode()

        if self.test_mode:
            self.logger.info(
                "Running in test mode - emails will be saved to files")
            self._setup_test_mode()
        else:
            self._validate_email_config()

    def _is_test_mode(self) -> bool:
        """Check if we're in test mode (file-based email)"""
        smtp_server = self.config.get('email', 'smtp_server', fallback='')
        smtp_port = self.config.get('email', 'smtp_port', fallback='25')

        return (
            smtp_server in ['test', 'file', 'debug'] or
            smtp_port == '0' or
            'test' in self.config.get(
                'email', 'from_address', fallback='').lower()
        )

    def _setup_test_mode(self):
        """Setup file-based email testing"""
        self.email_dir = Path("email_logs")
        self.email_dir.mkdir(exist_ok=True)
        self.logger.info(
            f"Test mode: emails will be saved to {self.email_dir}")

    def _validate_email_config(self):
        """Validate email configuration"""
        required_fields = ['smtp_server', 'smtp_port',
                           'from_address', 'to_addresses']

        for field in required_fields:
            if not self.config.get('email', field, fallback=''):
                raise ValueError(f"Email configuration missing: {field}")

        try:
            self._test_smtp_connection()
            self.logger.info("SMTP configuration validated")
        except Exception as e:
            self.logger.warning(f"SMTP connection test failed: {e}")

    def _test_smtp_connection(self):
        """Test SMTP connection"""
        smtp_server = self.config.get('email', 'smtp_server')
        smtp_port = int(self.config.get('email', 'smtp_port'))
        use_tls = self.config.getboolean('email', 'use_tls', fallback=False)

        with smtplib.SMTP(smtp_server, smtp_port, timeout=10) as server:
            if use_tls:
                server.starttls()

            username = self.config.get('email', 'username', fallback='')
            password = self.config.get('email', 'password', fallback='')

            if username and password:
                server.login(username, password)

    def format_earthquake_email(self, event: Dict, bulletin_text: str) -> tuple:
        """Format earthquake email using scbulletin output"""
        
        try:
            magnitude = event.get('Magnitude', 'Unknown')
            mag_type = event.get('MagType', '')
            location = event.get('EventLocationName', 'Unknown location')
            event_id = event.get('EventID', '')
            event_time = event.get('Time', '')
            
            # Parse and format event time for subject
            time_str = ""
            if event_time:
                try:
                    dt = datetime.fromisoformat(event_time.replace('Z', '+00:00'))
                    time_str = dt.strftime("%Y-%m-%d %H:%M UTC")
                except Exception as e:
                    self.logger.debug(f"Could not parse event time '{event_time}': {e}")
                    if 'T' in event_time:
                        date_part = event_time.split('T')[0]
                        time_part = event_time.split('T')[1][:5] if len(event_time.split('T')) > 1 else ""
                        time_str = f"{date_part} {time_part} UTC"
            
            # Format event ID for subject
            short_event_id = ""
            if event_id:
                if "/" in event_id:
                    short_event_id = event_id.split("/")[-1]
                else:
                    short_event_id = event_id
                if len(short_event_id) > 15:
                    short_event_id = short_event_id[:12] + "..."
            
            # Create subject line
            if isinstance(magnitude, (int, float)):
                mag_str = f"M{magnitude:.1f}"
            else:
                mag_str = "M?"
            
            # Build subject with datetime and event ID
            subject_parts = []
            
            # Add urgency indicator for larger earthquakes
            if isinstance(magnitude, (int, float)) and magnitude >= 5.0:
                subject_parts.append("🔴 URGENT")
            elif isinstance(magnitude, (int, float)) and magnitude >= 4.0:
                subject_parts.append("🟡")
            
            subject_parts.append(f"🚨 {mag_str} {mag_type}")
            
            if time_str:
                subject_parts.append(f"@ {time_str}")
            
            subject_parts.append(f"- {location}")
            
            if short_event_id:
                subject_parts.append(f"[{short_event_id}]")
            
            # Add region information if available
            matched_region = event.get('MatchedRegion')
            if matched_region:
                subject_parts.append(f"({matched_region})")
            
            distance_from_center = event.get('DistanceFromCenter')
            if distance_from_center is not None:
                subject_parts.append(f"({distance_from_center}km)")
            
            subject = " ".join(subject_parts)
            
            # Create Google Maps URL
            lat = event.get('Latitude', 0)
            lon = event.get('Longitude', 0)
            depth = event.get('Depth', 0)
            google_maps_url = f"https://maps.google.com/maps?q={lat},{lon}&z=8"
            
            # Create email body with bulletin
            current_time = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
            
            body = f"""EARTHQUAKE NOTIFICATION
{'=' * 50}

EVENT ID: {event_id}
Generated: {current_time}
System: SeisComP Native Toolchain + Multi-region

LOCATION: {lat:.4f}°, {lon:.4f}° (Depth: {depth:.1f}km)
GOOGLE MAPS: {google_maps_url}"""

            # Add region information if available
            if matched_region:
                region_min_mag = event.get('RegionMinMagnitude', 'N/A')
                body += f"""

REGION: {matched_region}
Region Min Magnitude: {region_min_mag}"""

            body += f"""

{'=' * 50}
OFFICIAL SEISMIC BULLETIN
{'=' * 50}

{bulletin_text}

{'=' * 50}
ATTACHMENTS
{'=' * 50}

- earthquake_map.png - Event location map with stations
- travel_time_curves.png - P/S wave travel time analysis (if available)
- waveforms.png - Recorded seismic waveforms (if available)
- event_location.kml - Google Earth/Maps compatible location file

{'=' * 50}
TECHNICAL INFORMATION
{'=' * 50}

- Data Source: SeisComP Database
- Bulletin Generator: scbulletin (-3 -k -e)
- Map Generator: scmapcut with complete event data
- KML Generator: scbulletin (--kml)
- Travel Time Analysis: ObsPy + TauP (IASP91 model)
- Waveform Data: FDSN Web Services (Localhost FDSNWS service)
- Processing: Automated earthquake monitoring system with multi-region filtering

For technical support or additional information, contact comoglu@gmail.com.
"""
            
            return subject, body
            
        except Exception as e:
            self.logger.error(f"Error formatting email: {e}")
            return (
                f"Earthquake Alert - {event.get('EventID', 'Unknown')}",
                f"An error occurred formatting the earthquake notification: {e}\n\nRaw event data:\n{event}\n\nBulletin text:\n{bulletin_text}"
            )

    def send_notification(self, event: Dict, bulletin_text: str, 
                        map_file: Optional[str] = None, 
                        curves_file: Optional[str] = None,
                        waveforms_file: Optional[str] = None,
                        kml_file: Optional[str] = None) -> bool:
        """Send earthquake notification email with multiple attachments"""
        
        if self.test_mode:
            return self._save_email_to_file(event, bulletin_text, map_file, curves_file, waveforms_file, kml_file)
        else:
            return self._send_email_smtp(event, bulletin_text, map_file, curves_file, waveforms_file, kml_file)

    def _save_email_to_file(self, event: Dict, bulletin_text: str,
                            map_file: Optional[str] = None,
                            curves_file: Optional[str] = None,
                            waveforms_file: Optional[str] = None,
                            kml_file: Optional[str] = None) -> bool:

        try:
            subject, body = self.format_earthquake_email(event, bulletin_text)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            event_id = event.get('EventID', 'unknown').replace('/', '_')
            filename = f"earthquake_email_{timestamp}_{event_id}.txt"
            filepath = self.email_dir / filename

            to_addresses = self.config.get(
                'email', 'to_addresses', fallback='test@test.local')
            from_address = self.config.get(
                'email', 'from_address', fallback='earthquake@test.local')

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"EARTHQUAKE EMAIL NOTIFICATION - TEST MODE\n")
                f.write(f"{'=' * 60}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"From: {from_address}\n")
                f.write(f"To: {to_addresses}\n")
                f.write(f"Subject: {subject}\n")

                attachments = []
                if map_file and Path(map_file).exists():
                    map_size = Path(map_file).stat().st_size
                    attachments.append(f"earthquake_map.png ({map_size} bytes)")

                if curves_file and Path(curves_file).exists():
                    curves_size = Path(curves_file).stat().st_size
                    attachments.append(f"travel_time_curves.png ({curves_size} bytes)")

                if waveforms_file and Path(waveforms_file).exists():
                    waveforms_size = Path(waveforms_file).stat().st_size
                    attachments.append(f"waveforms.png ({waveforms_size} bytes)")

                if kml_file and Path(kml_file).exists():
                    kml_size = Path(kml_file).stat().st_size
                    attachments.append(f"event_location.kml ({kml_size} bytes)")

                if attachments:
                    f.write(f"Attachments: {', '.join(attachments)}\n")
                else:
                    f.write(f"Attachments: None\n")

                f.write(f"\n{'-' * 30} EMAIL BODY {'-' * 30}\n")
                f.write(body)
                f.write(f"\n{'-' * 70}\n")

            self.logger.info(f"📧 Email saved to file: {filepath}")
            print(f"📧 Test email saved: {filepath}")

            return True

        except Exception as e:
            self.logger.error(f"Error saving email to file: {e}")
            return False

    def _send_email_smtp(self, event: Dict, bulletin_text: str,
                        map_file: Optional[str] = None,
                        curves_file: Optional[str] = None,
                        waveforms_file: Optional[str] = None,
                        kml_file: Optional[str] = None) -> bool:

        try:
            subject, body = self.format_earthquake_email(event, bulletin_text)

            msg = MIMEMultipart('related')
            msg['From'] = self.config.get('email', 'from_address')
            msg['Subject'] = subject

            to_addresses = [addr.strip() for addr in
                            self.config.get('email', 'to_addresses').split(',')]
            msg['To'] = ', '.join(to_addresses)

            msg['X-Priority'] = '2'
            msg['X-MSMail-Priority'] = 'High'
            msg['Importance'] = 'High'

            msg.attach(MIMEText(body, 'plain', 'utf-8'))

            if map_file and Path(map_file).exists():
                try:
                    with open(map_file, 'rb') as f:
                        img_data = f.read()

                    img = MIMEImage(img_data, name='earthquake_map.png')
                    img.add_header('Content-Disposition',
                                   'attachment', filename='earthquake_map.png')
                    img.add_header('Content-ID', '<earthquake_map>')
                    msg.attach(img)

                    self.logger.info("Map attached to email")
                except Exception as e:
                    self.logger.warning(f"Could not attach map: {e}")

            if curves_file and Path(curves_file).exists():
                try:
                    with open(curves_file, 'rb') as f:
                        img_data = f.read()

                    img = MIMEImage(img_data, name='travel_time_curves.png')
                    img.add_header('Content-Disposition', 'attachment',
                                   filename='travel_time_curves.png')
                    img.add_header('Content-ID', '<travel_time_curves>')
                    msg.attach(img)

                    self.logger.info("Travel time curves attached to email")
                except Exception as e:
                    self.logger.warning(
                        f"Could not attach travel time curves: {e}")

            if waveforms_file and Path(waveforms_file).exists():
                try:
                    with open(waveforms_file, 'rb') as f:
                        img_data = f.read()

                    img = MIMEImage(img_data, name='waveforms.png')
                    img.add_header('Content-Disposition', 'attachment',
                                   filename='waveforms.png')
                    img.add_header('Content-ID', '<waveforms>')
                    msg.attach(img)

                    self.logger.info("Waveforms attached to email")
                except Exception as e:
                    self.logger.warning(f"Could not attach waveforms: {e}")

            if kml_file and Path(kml_file).exists():
                try:
                    with open(kml_file, 'r', encoding='utf-8') as f:
                        kml_data = f.read()
                    
                    kml_attachment = MIMEApplication(kml_data.encode('utf-8'), _subtype='vnd.google-earth.kml+xml')
                    kml_attachment.add_header('Content-Disposition', 'attachment', filename='event_location.kml')
                    msg.attach(kml_attachment)
                    
                    self.logger.info("KML file attached to email")
                except Exception as e:
                    self.logger.warning(f"Could not attach KML file: {e}")

            smtp_server = self.config.get('email', 'smtp_server')
            smtp_port = int(self.config.get('email', 'smtp_port'))
            use_tls = self.config.getboolean(
                'email', 'use_tls', fallback=False)
            username = self.config.get('email', 'username', fallback='')
            password = self.config.get('email', 'password', fallback='')

            with smtplib.SMTP(smtp_server, smtp_port, timeout=30) as server:
                if use_tls:
                    server.starttls()

                if username and password:
                    server.login(username, password)

                text = msg.as_string()
                server.sendmail(msg['From'], to_addresses, text)

            self.logger.info(
                f"Email sent successfully for event {event.get('EventID', 'Unknown')}")
            return True

        except smtplib.SMTPException as e:
            self.logger.error(f"SMTP error sending email: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error sending email: {e}")

        return False


class EarthquakeNotifier:
    """Production earthquake notification system using SeisComP native tools"""

    def __init__(self, config_file: str = "earthquake_notifier.conf"):
        self.config_file = config_file
        self.config = self._load_configuration()
        self.logger = ProductionLogger.setup_logging(
            self.config.get('logging', 'log_file',
                            fallback='logs/earthquake_notifier.log'),
            self.config.get('logging', 'log_level', fallback='INFO')
        )

        self.logger.info(
            f"Starting Earthquake Notifier v{__version__} (SeisComP Native + ObsPy + Multi-region)")

        self.fdsnws_monitor = FDSNWSMonitor(
            self.config.get('service', 'fdsnws_url'),
            timeout=int(self.config.get('service', 'timeout', fallback='30'))
        )
        
        # Initialize multi-region filter
        self.multi_region_filter = MultiRegionFilter(self.config, self.logger)
        
        self.seiscomp_tools = SeisCompToolchain(self.config)
        self.email_notifier = EmailNotifier(self.config)

        self.sent_events = self._load_sent_events()
        self.is_running = False

        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    def _load_configuration(self) -> configparser.ConfigParser:
        """Load and validate configuration"""

        config = configparser.ConfigParser()

        default_config = {
            'service': {
                'fdsnws_url': os.getenv('FDSNWS_RECORDSTREAM', 'http://localhost:8080'),
                'min_magnitude': '2.5',
                'check_interval': '30',
                'timeout': '30',
                'max_age_hours': '24'
            },
            'filtering': {
                'filter_type': 'bounding_box',
                'min_latitude': '37.46',
                'max_latitude': '42.85',
                'min_longitude': '22.88',
                'max_longitude': '29.94',
                'filter_description': 'Canakkale Region'
            },
            'email': {
                'smtp_server': 'smtp.gmail.com',
                'smtp_port': '587',
                'use_tls': 'true',
                'username': '',
                'password': '',
                'from_address': '',
                'to_addresses': ''
            },
            'seiscomp': {
                'seiscomp_root': '/opt/seiscomp',
                'scxmldump_path': '/opt/seiscomp/bin/scxmldump',
                'scmapcut_path': '/opt/seiscomp/bin/scmapcut',
                'scbulletin_path': '/opt/seiscomp/bin/scbulletin',
                'database_url': 'mysql://sysop:sysop@localhost/seiscomp'
            },
            'logging': {
                'log_file': 'logs/earthquake_notifier.log',
                'log_level': 'INFO'
            },
            'advanced': {
                'concurrent_processing': 'false',
                'retry_failed_emails': 'true',
                'cleanup_temp_files': 'true',
                'generate_travel_time_curves': 'true',
                'generate_waveform_plots': 'true',
                'waveform_stations_limit': '10'
            }
        }

        if not Path(self.config_file).exists():
            for section, options in default_config.items():
                config.add_section(section)
                for key, value in options.items():
                    config.set(section, key, value)

            Path(self.config_file).parent.mkdir(parents=True, exist_ok=True)

            with open(self.config_file, 'w') as f:
                config.write(f)

            print(f"Created default configuration: {self.config_file}")
            print("Please edit the configuration file before running.")
            return config

        config.read(self.config_file)

        for section in default_config:
            if not config.has_section(section):
                config.add_section(section)
                for key, value in default_config[section].items():
                    config.set(section, key, value)

        return config

    def _load_sent_events(self) -> Set[str]:
        """Load set of already processed event IDs"""
        sent_file = Path("data/sent_events.json")
        sent_file.parent.mkdir(exist_ok=True)

        if sent_file.exists():
            try:
                with open(sent_file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return set(data)
                    elif isinstance(data, dict):
                        cutoff = datetime.now(
                            timezone.utc) - timedelta(days=30)
                        current_events = set()
                        for event_id, timestamp_str in data.items():
                            try:
                                timestamp = datetime.fromisoformat(
                                    timestamp_str.replace('Z', '+00:00'))
                                if timestamp > cutoff:
                                    current_events.add(event_id)
                            except:
                                continue
                        return current_events
            except Exception as e:
                self.logger.warning(f"Could not load sent events: {e}")

        return set()

    def _save_sent_events(self):
        """Save set of processed event IDs with timestamps"""
        sent_file = Path("data/sent_events.json")
        sent_file.parent.mkdir(exist_ok=True)

        try:
            data = {}
            if sent_file.exists():
                try:
                    with open(sent_file, 'r') as f:
                        existing = json.load(f)
                        if isinstance(existing, dict):
                            data = existing
                except:
                    pass

            current_time = datetime.now(timezone.utc).isoformat()
            for event_id in self.sent_events:
                if event_id not in data:
                    data[event_id] = current_time

            with open(sent_file, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            self.logger.error(f"Could not save sent events: {e}")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info(
            f"Received signal {signum}, shutting down gracefully...")
        self.is_running = False

    def check_new_events(self) -> int:
        """Check for new earthquake events and send notifications"""

        try:
            end_time = datetime.now(timezone.utc)
            max_age_hours = int(self.config.get(
                'service', 'max_age_hours', fallback='24'))
            start_time = end_time - timedelta(hours=max_age_hours)

            start_str = start_time.strftime("%Y-%m-%dT%H:%M:%S") + "Z"
            end_str = end_time.strftime("%Y-%m-%dT%H:%M:%S") + "Z"

            # Use multi-region filter for minimum magnitude
            min_magnitude = self.multi_region_filter.get_global_min_magnitude()

            self.logger.info(
                f"Checking events: {start_str} to {end_str}, global min_mag: {min_magnitude}")

            # Log region information
            if self.multi_region_filter.regions:
                self.logger.info(f"Active regions: {len(self.multi_region_filter.regions)}")
                for region in self.multi_region_filter.regions:
                    self.logger.info(f"  - {region['name']}: {region['type']}, min_mag={region['min_magnitude']}")

            events = self.fdsnws_monitor.get_event_list(
                self.config,
                self.multi_region_filter,
                starttime=start_str,
                endtime=end_str,
                minmagnitude=min_magnitude
            )

            if not events:
                self.logger.debug("No events found")
                return 0

            self.logger.info(f"Found {len(events)} events after filtering")

            new_events = []
            for event in events:
                event_id = event.get('EventID', '')

                if event_id not in self.sent_events:
                    new_events.append(event)
                else:
                    self.logger.debug(f"Event already processed: {event_id}")

            if not new_events:
                self.logger.info("No new events to process")
                return 0

            self.logger.info(f"Processing {len(new_events)} new events")

            processed_count = 0
            for event in new_events:
                if self._process_single_event(event):
                    processed_count += 1

                    event_id = event.get('EventID', '')
                    if event_id:
                        self.sent_events.add(event_id)

            if processed_count > 0:
                self._save_sent_events()
                self.logger.info(
                    f"Successfully processed {processed_count} events")

            return processed_count

        except Exception as e:
            self.logger.error(f"Error checking events: {e}")
            return 0

    def _process_single_event(self, event: Dict) -> bool:
        """Process a single earthquake event using SeisComP toolchain"""

        event_id = event.get('EventID', 'Unknown')
        magnitude = event.get('Magnitude', 'Unknown')
        matched_region = event.get('MatchedRegion', 'Unknown')

        self.logger.info(f"Processing event {event_id} (M{magnitude}) from region: {matched_region}")

        xml_file = None
        map_file = None
        curves_file = None
        waveforms_file = None
        kml_file = None
        bulletin_text = None

        try:
            # Step 1: Extract complete event XML using scxmldump
            xml_file = self.seiscomp_tools.extract_event_xml(event_id)
            if not xml_file:
                self.logger.error(f"Failed to extract XML for event {event_id}")
                return False

            # Step 2: Generate KML file FIRST (before ObsPy touches the XML)
            try:
                kml_file = self.seiscomp_tools.generate_kml(xml_file, event_id)
                if kml_file:
                    self.logger.info(f"KML generated for {event_id}")
                else:
                    self.logger.warning(f"KML generation failed for {event_id}")
            except Exception as e:
                self.logger.error(f"KML generation error for {event_id}: {e}")

            # Step 3: Generate map using scmapcut
            try:
                map_file = self.seiscomp_tools.generate_map(xml_file, event_id)
                if map_file:
                    self.logger.info(f"Map generated for {event_id}")
                else:
                    self.logger.warning(f"Map generation failed for {event_id}")
            except Exception as e:
                self.logger.error(f"Map generation error for {event_id}: {e}")

            # Step 4: Generate bulletin using scbulletin
            try:
                bulletin_text = self.seiscomp_tools.generate_bulletin(xml_file, event_id)
                if bulletin_text:
                    self.logger.info(f"Bulletin generated for {event_id}")
                else:
                    self.logger.error(f"Bulletin generation failed for {event_id}")
                    return False
            except Exception as e:
                self.logger.error(f"Bulletin generation error for {event_id}: {e}")
                return False

            # Step 5: Generate travel time curves using ObsPy
            if self.config.getboolean('advanced', 'generate_travel_time_curves', fallback=True):
                try:
                    curves_file = self.seiscomp_tools.generate_travel_time_curves(xml_file, event_id)
                    if curves_file:
                        self.logger.info(f"Travel time curves generated for {event_id}")
                    else:
                        self.logger.warning(f"Travel time curves generation failed for {event_id}")
                except Exception as e:
                    self.logger.error(f"Travel time curves generation error for {event_id}: {e}")

            # Step 6: Generate waveform plots using ObsPy
            if self.config.getboolean('advanced', 'generate_waveform_plots', fallback=True):
                try:
                    waveforms_file = self.seiscomp_tools.generate_waveform_plots(xml_file, event_id)
                    if waveforms_file:
                        self.logger.info(f"Waveform plots generated for {event_id}")
                    else:
                        self.logger.warning(f"Waveform plots generation failed for {event_id}")
                except Exception as e:
                    self.logger.error(f"Waveform plots generation error for {event_id}: {e}")

            # Step 7: Send email notification with all attachments
            email_sent = False
            try:
                email_sent = self.email_notifier.send_notification(
                    event, bulletin_text, map_file, curves_file, waveforms_file, kml_file
                )
                if email_sent:
                    self.logger.info(f"Email sent successfully for {event_id}")
                else:
                    self.logger.error(f"Email sending failed for {event_id}")
            except Exception as e:
                self.logger.error(f"Email sending error for {event_id}: {e}")

            # Step 8: Cleanup temporary files if configured
            if self.config.getboolean('advanced', 'cleanup_temp_files', fallback=True):
                self.seiscomp_tools.cleanup_temp_files(
                    xml_file, map_file, curves_file, waveforms_file, kml_file
                )

            return email_sent

        except Exception as e:
            self.logger.error(f"Unexpected error processing event {event_id}: {e}")

            # Cleanup files in case of error
            if self.config.getboolean('advanced', 'cleanup_temp_files', fallback=True):
                self.seiscomp_tools.cleanup_temp_files(
                    xml_file, map_file, curves_file, waveforms_file, kml_file
                )

            return False

    def run_monitoring_loop(self):
        """Main monitoring loop"""
        
        self.logger.info("Starting earthquake monitoring loop")
        self.is_running = True
        
        check_interval = int(self.config.get('service', 'check_interval', fallback='60'))
        
        print(f"🌍 Earthquake Notifier v{__version__} Started")
        print(f"📡 Monitoring: {self.config.get('service', 'fdsnws_url')}")
        print(f"🔄 Check interval: {check_interval} seconds")
        print(f"📧 Email mode: {'Test (file)' if self.email_notifier.test_mode else 'SMTP'}")
        print("🚨 Waiting for earthquakes...")
        print("Press Ctrl+C to stop")
        
        error_count = 0
        max_errors = 5
        
        while self.is_running:
            try:
                start_time = time.time()
                
                processed_count = self.check_new_events()
                
                if processed_count > 0:
                    print(f"✅ Processed {processed_count} new earthquake(s)")
                    error_count = 0  # Reset error count on success
                
                elapsed_time = time.time() - start_time
                sleep_time = max(0, check_interval - elapsed_time)
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
            except KeyboardInterrupt:
                self.logger.info("Received keyboard interrupt")
                break
            except Exception as e:
                error_count += 1
                self.logger.error(f"Error in monitoring loop: {e}")
                
                if error_count >= max_errors:
                    self.logger.error(f"Too many consecutive errors ({max_errors}), stopping")
                    break
                
                # Wait before retrying
                time.sleep(min(60, check_interval))
        
        self.is_running = False
        self.logger.info("Earthquake monitoring stopped")
        print("🛑 Earthquake monitoring stopped")

    def run_single_check(self):
        """Run a single check for testing purposes"""
        
        self.logger.info("Running single event check")
        print(f"🔍 Running single check...")
        
        try:
            processed_count = self.check_new_events()
            
            if processed_count > 0:
                print(f"✅ Processed {processed_count} new earthquake(s)")
                self.logger.info(f"Single check completed: {processed_count} events processed")
            else:
                print("ℹ️  No new earthquakes found")
                self.logger.info("Single check completed: no new events")
                
            return processed_count
            
        except Exception as e:
            self.logger.error(f"Error in single check: {e}")
            print(f"❌ Error during check: {e}")
            return 0

    def test_configuration(self):
        """Test system configuration and connections"""
        
        print(f"🧪 Testing Earthquake Notifier v{__version__} Configuration")
        print("=" * 60)
        
        all_tests_passed = True
        
        # Test 1: FDSNWS Connection
        print("1. Testing FDSNWS connection...")
        try:
            # Test basic connectivity
            test_params = {
                'starttime': (datetime.now(timezone.utc) - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S") + "Z",
                'endtime': datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S") + "Z",
                'minmagnitude': 0.0,
                'maxresults': 1
            }
            
            events = self.fdsnws_monitor.get_event_list(
                self.config, 
                self.multi_region_filter,
                **test_params
            )
            
            print(f"   ✅ FDSNWS connection successful")
            print(f"   📡 Service: {self.config.get('service', 'fdsnws_url')}")
            
        except Exception as e:
            print(f"   ❌ FDSNWS connection failed: {e}")
            all_tests_passed = False
        
        # Test 2: SeisComP Tools
        print("\n2. Testing SeisComP tools...")
        try:
            self.seiscomp_tools._verify_tools()
            print("   ✅ All SeisComP tools verified")
        except Exception as e:
            print(f"   ❌ SeisComP tools verification failed: {e}")
            all_tests_passed = False
        
        # Test 3: Email Configuration
        print("\n3. Testing email configuration...")
        try:
            if self.email_notifier.test_mode:
                print("   ✅ Email test mode configured")
                print(f"   📁 Test emails will be saved to: {self.email_notifier.email_dir}")
            else:
                self.email_notifier._test_smtp_connection()
                print("   ✅ SMTP connection successful")
                print(f"   📧 Server: {self.config.get('email', 'smtp_server')}:{self.config.get('email', 'smtp_port')}")
        except Exception as e:
            print(f"   ❌ Email configuration failed: {e}")
            all_tests_passed = False
        
        # Test 4: Multi-region Filter
        print("\n4. Testing multi-region filter...")
        try:
            if self.multi_region_filter.regions:
                print(f"   ✅ Multi-region filter configured with {len(self.multi_region_filter.regions)} regions:")
                for region in self.multi_region_filter.regions:
                    print(f"      - {region['name']}: {region['type']}, min_mag={region['min_magnitude']}")
            else:
                print("   ℹ️  No multi-region filter configured (using legacy filtering)")
        except Exception as e:
            print(f"   ❌ Multi-region filter failed: {e}")
            all_tests_passed = False
        
        # Test 5: ObsPy Features
        print("\n5. Testing ObsPy features...")
        if OBSPY_AVAILABLE:
            print("   ✅ ObsPy available")
            if self.seiscomp_tools.curves_generator.available:
                print("   ✅ Travel time curves enabled")
            else:
                print("   ⚠️  Travel time curves disabled")
            
            if self.seiscomp_tools.waveform_plotter.available:
                print("   ✅ Waveform plots enabled")
            else:
                print("   ⚠️  Waveform plots disabled")
        else:
            print("   ⚠️  ObsPy not available - advanced features disabled")
        
        # Test 6: File Permissions
        print("\n6. Testing file permissions...")
        try:
            # Test log directory
            log_dir = Path(self.config.get('logging', 'log_file')).parent
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Test data directory
            data_dir = Path("data")
            data_dir.mkdir(exist_ok=True)
            
            # Test temp directory access
            temp_file = tempfile.NamedTemporaryFile(delete=True)
            temp_file.close()
            
            print("   ✅ File permissions OK")
            
        except Exception as e:
            print(f"   ❌ File permission error: {e}")
            all_tests_passed = False
        
        print("\n" + "=" * 60)
        if all_tests_passed:
            print("🎉 All tests passed! System ready for operation.")
        else:
            print("❌ Some tests failed. Please check configuration.")
        
        return all_tests_passed


def main():
    """Main entry point"""
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description=f"Earthquake Notifier v{__version__} - SeisComP Native + ObsPy + Multi-region",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run continuous monitoring
  python mceqnotifier.py --monitor
  
  # Run single check
  python mceqnotifier.py --check
  
  # Test configuration
  python mceqnotifier.py --test
  
  # Use custom config file
  python mceqnotifier.py --config myconfig.conf --monitor
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        default='earthquake_notifier.conf',
        help='Configuration file path (default: earthquake_notifier.conf)'
    )
    
    parser.add_argument(
        '--monitor', '-m',
        action='store_true',
        help='Run continuous monitoring loop'
    )
    
    parser.add_argument(
        '--check', '-k',
        action='store_true',
        help='Run single event check'
    )
    
    parser.add_argument(
        '--test', '-t',
        action='store_true',
        help='Test configuration and connections'
    )
    
    parser.add_argument(
        '--version', '-v',
        action='version',
        version=f'Earthquake Notifier v{__version__}'
    )
    
    args = parser.parse_args()
    
    # If no action specified, show help
    if not (args.monitor or args.check or args.test):
        parser.print_help()
        return 1
    
    try:
        # Initialize the notifier
        notifier = EarthquakeNotifier(args.config)
        
        if args.test:
            # Test configuration
            success = notifier.test_configuration()
            return 0 if success else 1
            
        elif args.check:
            # Single check
            processed = notifier.run_single_check()
            return 0
            
        elif args.monitor:
            # Continuous monitoring
            notifier.run_monitoring_loop()
            return 0
            
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")
        return 0
        
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        return 1
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
