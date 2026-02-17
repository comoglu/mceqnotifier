#!/usr/bin/env python3
"""
SeisComP Bulletin Generator Tool

A standalone CLI and GUI tool for querying seismic events by time range and event type,
then generating bulletins in multiple formats using SeisComP native tools.

Features:
- Batch mode: Query events by time range + event type, generate bulletins
- Real-time mode: Continuous monitoring for new events of specified types
- Output formats: Text bulletin (autoloc3), FDSNWS, KML, PNG maps
- Interfaces: CLI (argparse) + GUI (PyQt6)

Version: 1.0.0
Author: Mustafa Comoglu
"""

import argparse
import configparser
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import List, Dict, Optional, Set, Callable, Any

# Version and metadata
__version__ = "1.0.0"
__author__ = "Mustafa Comoglu"

# Supported SeisComP event types
EVENT_TYPES = [
    "earthquake",
    "explosion",
    "quarry blast",
    "chemical explosion",
    "nuclear explosion",
    "landslide",
    "debris avalanche",
    "rockslide",
    "mine collapse",
    "volcanic eruption",
    "meteor impact",
    "plane crash",
    "building collapse",
    "sonic boom",
    "other",
    "not existing",
    "not locatable",
    "outside of network interest",
    "duplicate",
]

# Default timeouts (seconds)
DEFAULT_TIMEOUT = 120
BULLETIN_TIMEOUT = 60

# UTF-8 environment for SeisComP subprocess calls
_UTF8_ENV = {**os.environ, 'LANG': 'C.UTF-8', 'LC_ALL': 'C.UTF-8'}


# =============================================================================
# Infrastructure Classes
# =============================================================================

class ProductionLogger:
    """Production-grade logging setup with rotation"""

    @staticmethod
    def setup_logging(log_file: str = "logs/bulletin_generator.log",
                      log_level: str = "INFO",
                      logger_name: str = "bulletin_generator") -> logging.Logger:
        """Setup comprehensive logging with file rotation and console output"""

        # Create logs directory if it doesn't exist
        log_path = Path(log_file).parent
        log_path.mkdir(parents=True, exist_ok=True)

        # Configure logging level
        numeric_level = getattr(logging, log_level.upper(), logging.INFO)

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        )

        # Setup logger
        logger = logging.getLogger(logger_name)
        logger.setLevel(numeric_level)

        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # File handler with rotation (10MB per file, 5 backups)
        try:
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


class ConfigManager:
    """Configuration management using INI format"""

    def __init__(self, config_file: str = "config.ini"):
        self.config_file = config_file
        self.config = configparser.ConfigParser()
        self._loaded = False

    def load(self) -> configparser.ConfigParser:
        """Load configuration from file"""
        if not Path(self.config_file).exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_file}")

        self.config.read(self.config_file)
        self._loaded = True
        return self.config

    def get_database_url(self) -> str:
        """Get SeisComP database connection URL"""
        if not self._loaded:
            self.load()
        return self.config.get('seiscomp', 'database_url',
                               fallback='mysql://sysop:sysop@localhost/seiscomp')

    def get_seiscomp_root(self) -> str:
        """Get SeisComP installation root directory"""
        if not self._loaded:
            self.load()
        return self.config.get('seiscomp', 'seiscomp_root', fallback='/opt/seiscomp')

    def get_tool_path(self, tool_name: str) -> str:
        """Get path to a SeisComP tool"""
        if not self._loaded:
            self.load()

        seiscomp_root = self.get_seiscomp_root()
        default_path = f'{seiscomp_root}/bin/{tool_name}'
        return self.config.get('seiscomp', f'{tool_name}_path', fallback=default_path)

    def get_seiscomp_paths(self) -> Dict[str, str]:
        """Get all SeisComP tool paths"""
        return {
            'scevtls': self.get_tool_path('scevtls'),
            'scxmldump': self.get_tool_path('scxmldump'),
            'scbulletin': self.get_tool_path('scbulletin'),
            'scmapcut': self.get_tool_path('scmapcut'),
        }

    def get_output_directory(self) -> str:
        """Get default output directory"""
        if not self._loaded:
            self.load()
        return self.config.get('bulletin_generator', 'output_dir', fallback='output')

    def get_check_interval(self) -> int:
        """Get real-time monitoring check interval in seconds"""
        if not self._loaded:
            self.load()
        return self.config.getint('bulletin_generator', 'check_interval', fallback=60)

    def get_lookback_hours(self) -> int:
        """Get initial lookback hours for real-time monitoring"""
        if not self._loaded:
            self.load()
        return self.config.getint('bulletin_generator', 'lookback_hours', fallback=1)

    def get_map_dimensions(self) -> str:
        """Get map image dimensions"""
        if not self._loaded:
            self.load()
        return self.config.get('bulletin_generator', 'map_dimensions', fallback='1024x768')

    def get_map_min_magnitude(self) -> float:
        """Get minimum magnitude for map station display"""
        if not self._loaded:
            self.load()
        return self.config.getfloat('bulletin_generator', 'map_min_magnitude', fallback=3.0)

    def should_cleanup_xml(self) -> bool:
        """Check if XML files should be cleaned up after processing"""
        if not self._loaded:
            self.load()
        return self.config.getboolean('bulletin_generator', 'cleanup_xml', fallback=True)


class TempFileManager:
    """Manage temporary files with automatic cleanup"""

    def __init__(self, cleanup_on_exit: bool = True):
        self.cleanup_on_exit = cleanup_on_exit
        self.temp_files: List[Path] = []
        self._lock = threading.Lock()

    def create_temp_file(self, suffix: str = '', prefix: str = 'bulletin_') -> str:
        """Create a temporary file and track it for cleanup"""
        with self._lock:
            temp_file = tempfile.NamedTemporaryFile(
                mode='w+', suffix=suffix, delete=False, prefix=prefix
            )
            temp_file.close()
            self.temp_files.append(Path(temp_file.name))
            return temp_file.name

    def cleanup_file(self, file_path: str) -> None:
        """Clean up a specific file"""
        path = Path(file_path)
        with self._lock:
            if path in self.temp_files:
                self.temp_files.remove(path)
        try:
            if path.exists():
                path.unlink()
        except Exception:
            pass

    def cleanup_all(self) -> None:
        """Clean up all tracked temporary files"""
        with self._lock:
            for path in self.temp_files[:]:
                try:
                    if path.exists():
                        path.unlink()
                except Exception:
                    pass
            self.temp_files.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cleanup_on_exit:
            self.cleanup_all()
        return False


# =============================================================================
# SeisComP Tool Wrappers
# =============================================================================

class SeisCompToolError(Exception):
    """Custom exception for SeisComP tool failures"""

    def __init__(self, tool: str, returncode: int, stderr: str, stdout: str = ''):
        self.tool = tool
        self.returncode = returncode
        self.stderr = stderr
        self.stdout = stdout
        super().__init__(f"{tool} failed (code {returncode}): {stderr}")


class ScEvtLs:
    """Wrapper for scevtls - list events by time range and type"""

    def __init__(self, tool_path: str, database_url: str, logger: logging.Logger):
        self.tool_path = tool_path
        self.database_url = database_url
        self.logger = logger
        self._verify_tool()

    def _verify_tool(self) -> None:
        """Verify scevtls is available"""
        if not Path(self.tool_path).exists():
            raise FileNotFoundError(f"scevtls not found at {self.tool_path}")
        self.logger.info(f"scevtls available at {self.tool_path}")

    def list_events(
        self,
        begin: datetime,
        end: datetime,
        event_type: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT
    ) -> List[str]:
        """
        Query events from SeisComP database by time range and optionally event type.

        Args:
            begin: Start time (UTC)
            end: End time (UTC)
            event_type: Optional event type filter (e.g., "earthquake", "quarry blast")
            timeout: Command timeout in seconds

        Returns:
            List of event IDs
        """
        # Format times for scevtls (ISO format)
        begin_str = begin.strftime('%Y-%m-%dT%H:%M:%S')
        end_str = end.strftime('%Y-%m-%dT%H:%M:%S')

        cmd = [
            self.tool_path,
            '-d', self.database_url,
            '--begin', begin_str,
            '--end', end_str,
        ]

        if event_type:
            cmd.extend(['--event-type', event_type])

        self.logger.info(f"Querying events: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=_UTF8_ENV
            )

            if result.returncode != 0:
                raise SeisCompToolError('scevtls', result.returncode, result.stderr, result.stdout)

            return self._parse_output(result.stdout)

        except subprocess.TimeoutExpired:
            self.logger.error(f"scevtls timeout after {timeout}s")
            raise

    def _parse_output(self, stdout: str) -> List[str]:
        """Parse event IDs from scevtls output (one ID per line)"""
        event_ids = []
        for line in stdout.strip().split('\n'):
            line = line.strip()
            if line:
                event_ids.append(line)
        self.logger.info(f"Found {len(event_ids)} events")
        return event_ids


class ScXmlDump:
    """Wrapper for scxmldump - extract event XML from database"""

    def __init__(self, tool_path: str, database_url: str, logger: logging.Logger):
        self.tool_path = tool_path
        self.database_url = database_url
        self.logger = logger
        self._verify_tool()

    def _verify_tool(self) -> None:
        """Verify scxmldump is available"""
        if not Path(self.tool_path).exists():
            raise FileNotFoundError(f"scxmldump not found at {self.tool_path}")
        self.logger.info(f"scxmldump available at {self.tool_path}")

    def extract(
        self,
        event_id: str,
        output_file: Optional[str] = None,
        include_picks: bool = True,
        include_amplitudes: bool = True,
        include_magnitudes: bool = True,
        include_focal_mechanisms: bool = True,
        formatted: bool = True,
        preferred_only: bool = False,
        timeout: int = DEFAULT_TIMEOUT
    ) -> str:
        """
        Extract complete event XML using scxmldump.

        Args:
            event_id: SeisComP event ID
            output_file: Output file path (auto-generated if None)
            include_picks: Include pick data (-P)
            include_amplitudes: Include amplitude data (-A)
            include_magnitudes: Include magnitude data (-M)
            include_focal_mechanisms: Include focal mechanism data (-F)
            formatted: Pretty-print XML output (-f)
            preferred_only: Only export preferred origin (-p)
            timeout: Command timeout in seconds

        Returns:
            Path to the generated XML file
        """
        # Generate output file if not specified
        if output_file is None:
            safe_id = event_id.replace('/', '_').replace(':', '_')
            output_file = tempfile.NamedTemporaryFile(
                mode='w+', suffix='.xml', delete=False,
                prefix=f'event_{safe_id}_'
            ).name

        # Build command flags
        flags = ''
        if include_picks:
            flags += 'P'
        if include_amplitudes:
            flags += 'A'
        if include_magnitudes:
            flags += 'M'
        if include_focal_mechanisms:
            flags += 'F'
        if formatted:
            flags += 'f'
        if preferred_only:
            flags += 'p'

        cmd = [
            self.tool_path,
            f'-{flags}' if flags else '',
            '-d', self.database_url,
            '-E', event_id,
            '-o', output_file
        ]
        # Remove empty strings from command
        cmd = [c for c in cmd if c]

        self.logger.info(f"Extracting event XML: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=_UTF8_ENV
            )

            if result.returncode != 0:
                # Clean up failed output file
                try:
                    Path(output_file).unlink()
                except:
                    pass
                raise SeisCompToolError('scxmldump', result.returncode, result.stderr, result.stdout)

            # Verify output file exists and has content
            output_path = Path(output_file)
            if not output_path.exists() or output_path.stat().st_size == 0:
                raise SeisCompToolError('scxmldump', 0, 'Output file is empty or missing')

            file_size = output_path.stat().st_size
            self.logger.info(f"Event XML extracted ({file_size} bytes): {output_file}")
            return output_file

        except subprocess.TimeoutExpired:
            self.logger.error(f"scxmldump timeout for event {event_id}")
            try:
                Path(output_file).unlink()
            except:
                pass
            raise


class ScBulletin:
    """Wrapper for scbulletin - generate bulletins in multiple formats"""

    def __init__(self, tool_path: str, logger: logging.Logger):
        self.tool_path = tool_path
        self.logger = logger
        self._verify_tool()

    def _verify_tool(self) -> None:
        """Verify scbulletin is available"""
        if not Path(self.tool_path).exists():
            raise FileNotFoundError(f"scbulletin not found at {self.tool_path}")
        self.logger.info(f"scbulletin available at {self.tool_path}")

    def generate_autoloc3(
        self,
        xml_file: str,
        enhanced: bool = True,
        km_distances: bool = True,
        output_file: Optional[str] = None,
        timeout: int = BULLETIN_TIMEOUT
    ) -> str:
        """
        Generate bulletin in autoloc3 format.

        Args:
            xml_file: Input SeisComP XML file
            enhanced: Use enhanced output with higher precision (-e)
            km_distances: Print distances in km instead of degrees (-k)
            output_file: Output file path (returns stdout if None)
            timeout: Command timeout in seconds

        Returns:
            Bulletin text (if output_file is None) or path to output file
        """
        cmd = [self.tool_path, '-i', xml_file, '-3']

        if enhanced:
            cmd.append('-e')
        if km_distances:
            cmd.append('-k')
        if output_file:
            cmd.extend(['-o', output_file])

        self.logger.info(f"Generating autoloc3 bulletin: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=_UTF8_ENV
            )

            if result.returncode != 0:
                raise SeisCompToolError('scbulletin', result.returncode, result.stderr, result.stdout)

            if output_file:
                self.logger.info(f"Autoloc3 bulletin saved to: {output_file}")
                return output_file
            else:
                self.logger.info("Autoloc3 bulletin generated successfully")
                return result.stdout

        except subprocess.TimeoutExpired:
            self.logger.error("scbulletin timeout generating autoloc3 format")
            raise

    def generate_fdsnws(
        self,
        xml_file: str,
        extra_columns: bool = False,
        output_file: Optional[str] = None,
        timeout: int = BULLETIN_TIMEOUT
    ) -> str:
        """
        Generate bulletin in FDSNWS event text format.

        Args:
            xml_file: Input SeisComP XML file
            extra_columns: Include additional columns (-x)
            output_file: Output file path (returns stdout if None)
            timeout: Command timeout in seconds

        Returns:
            FDSNWS text (if output_file is None) or path to output file
        """
        cmd = [self.tool_path, '-i', xml_file, '-4']

        if extra_columns:
            cmd.append('-x')
        if output_file:
            cmd.extend(['-o', output_file])

        self.logger.info(f"Generating FDSNWS format: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=_UTF8_ENV
            )

            if result.returncode != 0:
                raise SeisCompToolError('scbulletin', result.returncode, result.stderr, result.stdout)

            if output_file:
                self.logger.info(f"FDSNWS format saved to: {output_file}")
                return output_file
            else:
                self.logger.info("FDSNWS format generated successfully")
                return result.stdout

        except subprocess.TimeoutExpired:
            self.logger.error("scbulletin timeout generating FDSNWS format")
            raise

    def generate_kml(
        self,
        xml_file: str,
        output_file: str,
        timeout: int = BULLETIN_TIMEOUT
    ) -> str:
        """
        Generate KML file for Google Earth.

        Args:
            xml_file: Input SeisComP XML file
            output_file: Output KML file path (required)
            timeout: Command timeout in seconds

        Returns:
            Path to the generated KML file
        """
        cmd = [self.tool_path, '-i', xml_file, '--kml', '-o', output_file]

        self.logger.info(f"Generating KML: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=_UTF8_ENV
            )

            if result.returncode != 0:
                raise SeisCompToolError('scbulletin', result.returncode, result.stderr, result.stdout)

            # Verify output file exists and has content
            output_path = Path(output_file)
            if not output_path.exists() or output_path.stat().st_size == 0:
                raise SeisCompToolError('scbulletin', 0, 'KML output file is empty or missing')

            file_size = output_path.stat().st_size
            self.logger.info(f"KML generated ({file_size} bytes): {output_file}")
            return output_file

        except subprocess.TimeoutExpired:
            self.logger.error("scbulletin timeout generating KML")
            raise


class ScMapCut:
    """Wrapper for scmapcut - generate map images"""

    def __init__(self, tool_path: str, logger: logging.Logger):
        self.tool_path = tool_path
        self.logger = logger
        self._verify_tool()

    def _verify_tool(self) -> None:
        """Verify scmapcut is available"""
        if not Path(self.tool_path).exists():
            raise FileNotFoundError(f"scmapcut not found at {self.tool_path}")
        self.logger.info(f"scmapcut available at {self.tool_path}")

    def generate(
        self,
        event_id: str,
        xml_file: str,
        output_file: str,
        dimensions: str = "1024x768",
        min_magnitude: float = 3.0,
        timeout: int = DEFAULT_TIMEOUT
    ) -> str:
        """
        Generate map image using scmapcut.

        Args:
            event_id: SeisComP event ID
            xml_file: Event parameters XML file
            output_file: Output PNG file path
            dimensions: Image dimensions (e.g., "1024x768")
            min_magnitude: Minimum magnitude for station display
            timeout: Command timeout in seconds

        Returns:
            Path to the generated PNG file
        """
        cmd = [
            self.tool_path,
            '-E', event_id,
            '--ep', xml_file,
            '-m', str(min_magnitude),
            '-d', dimensions,
            '-o', output_file
        ]

        self.logger.info(f"Generating map: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=_UTF8_ENV
            )

            if result.returncode != 0:
                raise SeisCompToolError('scmapcut', result.returncode, result.stderr, result.stdout)

            # Verify output file exists and has content
            output_path = Path(output_file)
            if not output_path.exists() or output_path.stat().st_size == 0:
                raise SeisCompToolError('scmapcut', 0, 'Map output file is empty or missing')

            file_size = output_path.stat().st_size
            self.logger.info(f"Map generated ({file_size} bytes): {output_file}")
            return output_file

        except subprocess.TimeoutExpired:
            self.logger.error(f"scmapcut timeout for event {event_id}")
            raise


# =============================================================================
# Business Logic Classes
# =============================================================================

@dataclass
class BulletinOutputConfig:
    """Configuration for bulletin output formats"""
    autoloc3: bool = True       # -3 -e -k format
    fdsnws: bool = True         # -4 format
    kml: bool = True            # --kml format
    map_image: bool = True      # scmapcut PNG
    keep_xml: bool = False      # Keep intermediate XML file
    combined: bool = False      # Combine all autoloc3 bulletins into one file


@dataclass
class BulletinOutput:
    """Container for generated bulletin outputs"""
    event_id: str
    autoloc3_text: Optional[str] = None
    autoloc3_file: Optional[str] = None
    fdsnws_text: Optional[str] = None
    fdsnws_file: Optional[str] = None
    kml_file: Optional[str] = None
    map_file: Optional[str] = None
    xml_file: Optional[str] = None
    error: Optional[str] = None
    processing_time: float = 0.0

    def is_success(self) -> bool:
        """Check if bulletin was generated successfully (at least partially)"""
        return self.error is None and (
            self.autoloc3_text is not None or
            self.autoloc3_file is not None or
            self.fdsnws_text is not None or
            self.kml_file is not None or
            self.map_file is not None
        )


class BulletinGenerator:
    """Core bulletin generation orchestrator"""

    def __init__(
        self,
        config: ConfigManager,
        logger: logging.Logger,
        temp_manager: Optional[TempFileManager] = None
    ):
        self.config = config
        self.logger = logger
        self.temp_manager = temp_manager or TempFileManager(cleanup_on_exit=False)

        # Initialize tool wrappers
        paths = config.get_seiscomp_paths()
        db_url = config.get_database_url()

        self.scxmldump = ScXmlDump(paths['scxmldump'], db_url, logger)
        self.scbulletin = ScBulletin(paths['scbulletin'], logger)
        self.scmapcut = ScMapCut(paths['scmapcut'], logger)

        # Map settings
        self.map_dimensions = config.get_map_dimensions()
        self.map_min_magnitude = config.get_map_min_magnitude()

    def generate_bulletin(
        self,
        event_id: str,
        output_config: BulletinOutputConfig,
        output_dir: Optional[str] = None
    ) -> BulletinOutput:
        """
        Generate complete bulletin for a single event.

        Args:
            event_id: SeisComP event ID
            output_config: Configuration specifying which outputs to generate
            output_dir: Directory to save output files (uses temp files if None)

        Returns:
            BulletinOutput containing all generated outputs
        """
        start_time = time.time()
        result = BulletinOutput(event_id=event_id)
        xml_file = None

        try:
            # Create output directory for this event if specified
            if output_dir:
                safe_id = event_id.replace('/', '_').replace(':', '_')
                event_output_dir = Path(output_dir) / safe_id
                event_output_dir.mkdir(parents=True, exist_ok=True)
            else:
                event_output_dir = None

            # Step 1: Extract event XML
            self.logger.info(f"Processing event: {event_id}")
            xml_file = self.scxmldump.extract(event_id)

            # Step 2: Generate requested outputs
            if output_config.autoloc3:
                try:
                    if event_output_dir:
                        output_file = str(event_output_dir / 'bulletin_autoloc3.txt')
                        self.scbulletin.generate_autoloc3(xml_file, output_file=output_file)
                        result.autoloc3_file = output_file
                        # Also store text content
                        with open(output_file, 'r') as f:
                            result.autoloc3_text = f.read()
                    else:
                        result.autoloc3_text = self.scbulletin.generate_autoloc3(xml_file)
                except Exception as e:
                    self.logger.error(f"Failed to generate autoloc3: {e}")

            if output_config.fdsnws:
                try:
                    if event_output_dir:
                        output_file = str(event_output_dir / 'bulletin_fdsnws.txt')
                        self.scbulletin.generate_fdsnws(xml_file, output_file=output_file)
                        result.fdsnws_file = output_file
                        with open(output_file, 'r') as f:
                            result.fdsnws_text = f.read()
                    else:
                        result.fdsnws_text = self.scbulletin.generate_fdsnws(xml_file)
                except Exception as e:
                    self.logger.error(f"Failed to generate FDSNWS: {e}")

            if output_config.kml:
                try:
                    if event_output_dir:
                        output_file = str(event_output_dir / 'event.kml')
                    else:
                        output_file = self.temp_manager.create_temp_file(suffix='.kml')
                    self.scbulletin.generate_kml(xml_file, output_file)
                    result.kml_file = output_file
                except Exception as e:
                    self.logger.error(f"Failed to generate KML: {e}")

            if output_config.map_image:
                try:
                    if event_output_dir:
                        output_file = str(event_output_dir / 'map.png')
                    else:
                        output_file = self.temp_manager.create_temp_file(suffix='.png')
                    self.scmapcut.generate(
                        event_id, xml_file, output_file,
                        dimensions=self.map_dimensions,
                        min_magnitude=self.map_min_magnitude
                    )
                    result.map_file = output_file
                except Exception as e:
                    self.logger.error(f"Failed to generate map: {e}")

            # Handle XML file
            if output_config.keep_xml and event_output_dir:
                xml_dest = str(event_output_dir / 'event.xml')
                shutil.copy(xml_file, xml_dest)
                result.xml_file = xml_dest

        except Exception as e:
            result.error = str(e)
            self.logger.error(f"Error processing event {event_id}: {e}")

        finally:
            # Clean up temporary XML file
            if xml_file and not output_config.keep_xml:
                try:
                    Path(xml_file).unlink()
                except:
                    pass

        result.processing_time = time.time() - start_time
        self.logger.info(f"Event {event_id} processed in {result.processing_time:.2f}s")
        return result

    def generate_bulletins_batch(
        self,
        event_ids: List[str],
        output_config: BulletinOutputConfig,
        output_dir: str,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> List[BulletinOutput]:
        """
        Generate bulletins for multiple events.

        Args:
            event_ids: List of event IDs to process
            output_config: Configuration specifying which outputs to generate
            output_dir: Directory to save output files
            progress_callback: Optional callback for progress updates (current, total, event_id)

        Returns:
            List of BulletinOutput for each event
        """
        results = []
        total = len(event_ids)

        for i, event_id in enumerate(event_ids, 1):
            if progress_callback:
                progress_callback(i, total, event_id)

            result = self.generate_bulletin(event_id, output_config, output_dir)
            results.append(result)

        return results


class BatchProcessor:
    """Batch processing for time-range based bulletin generation"""

    def __init__(self, config: ConfigManager, logger: logging.Logger):
        self.config = config
        self.logger = logger

        # Initialize tools
        paths = config.get_seiscomp_paths()
        db_url = config.get_database_url()

        self.scevtls = ScEvtLs(paths['scevtls'], db_url, logger)
        self.generator = BulletinGenerator(config, logger)

    def query_events(
        self,
        begin: datetime,
        end: datetime,
        event_types: Optional[List[str]] = None
    ) -> List[str]:
        """
        Query events by time range and optional event types.

        Args:
            begin: Start time (UTC)
            end: End time (UTC)
            event_types: List of event types to filter (queries all if None or empty)

        Returns:
            List of event IDs
        """
        all_event_ids = set()

        if event_types:
            # Query each event type separately
            for event_type in event_types:
                try:
                    ids = self.scevtls.list_events(begin, end, event_type)
                    all_event_ids.update(ids)
                except Exception as e:
                    self.logger.error(f"Error querying events of type '{event_type}': {e}")
        else:
            # Query all events (no type filter)
            try:
                ids = self.scevtls.list_events(begin, end)
                all_event_ids.update(ids)
            except Exception as e:
                self.logger.error(f"Error querying events: {e}")

        return sorted(all_event_ids)

    def process_time_range(
        self,
        begin: datetime,
        end: datetime,
        event_types: Optional[List[str]] = None,
        output_config: Optional[BulletinOutputConfig] = None,
        output_dir: str = "output",
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> List[BulletinOutput]:
        """
        Process all events in a time range.

        Args:
            begin: Start time (UTC)
            end: End time (UTC)
            event_types: Optional list of event types to filter
            output_config: Bulletin output configuration
            output_dir: Output directory
            progress_callback: Progress callback function

        Returns:
            List of BulletinOutput for each processed event
        """
        if output_config is None:
            output_config = BulletinOutputConfig()

        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Query events
        self.logger.info(f"Querying events from {begin} to {end}")
        if event_types:
            self.logger.info(f"Filtering by event types: {event_types}")

        event_ids = self.query_events(begin, end, event_types)

        if not event_ids:
            self.logger.info("No events found matching criteria")
            return []

        self.logger.info(f"Found {len(event_ids)} events to process")

        # Generate bulletins
        return self.generator.generate_bulletins_batch(
            event_ids, output_config, output_dir, progress_callback
        )


class RealTimeMonitor:
    """Real-time monitoring for new events"""

    def __init__(
        self,
        config: ConfigManager,
        logger: logging.Logger,
        check_interval: Optional[int] = None
    ):
        self.config = config
        self.logger = logger
        self.check_interval = check_interval or config.get_check_interval()

        # Initialize processor
        self.processor = BatchProcessor(config, logger)

        # State
        self.processed_events: Set[str] = set()
        self.is_running: bool = False
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Callbacks
        self._event_callback: Optional[Callable[[BulletinOutput], None]] = None
        self._status_callback: Optional[Callable[[str], None]] = None

    def start(
        self,
        event_types: Optional[List[str]] = None,
        output_config: Optional[BulletinOutputConfig] = None,
        output_dir: str = "output",
        lookback_hours: Optional[int] = None,
        event_callback: Optional[Callable[[BulletinOutput], None]] = None,
        status_callback: Optional[Callable[[str], None]] = None
    ) -> None:
        """
        Start real-time monitoring in a background thread.

        Args:
            event_types: Event types to monitor (all if None)
            output_config: Bulletin output configuration
            output_dir: Output directory
            lookback_hours: Initial lookback period in hours
            event_callback: Called when a new event is processed
            status_callback: Called with status updates
        """
        if self.is_running:
            self.logger.warning("Monitor is already running")
            return

        self._event_callback = event_callback
        self._status_callback = status_callback

        if output_config is None:
            output_config = BulletinOutputConfig()
        if lookback_hours is None:
            lookback_hours = self.config.get_lookback_hours()

        self._stop_event.clear()
        self.is_running = True

        self._thread = threading.Thread(
            target=self._monitoring_loop,
            args=(event_types, output_config, output_dir, lookback_hours),
            daemon=True
        )
        self._thread.start()
        self.logger.info("Real-time monitoring started")

    def stop(self) -> None:
        """Stop real-time monitoring"""
        if not self.is_running:
            return

        self.logger.info("Stopping real-time monitoring...")
        self._stop_event.set()

        if self._thread:
            self._thread.join(timeout=10)

        self.is_running = False
        self.logger.info("Real-time monitoring stopped")

    def get_status(self) -> Dict[str, Any]:
        """Get current monitoring status"""
        with self._lock:
            return {
                'is_running': self.is_running,
                'processed_count': len(self.processed_events),
                'check_interval': self.check_interval,
            }

    def _monitoring_loop(
        self,
        event_types: Optional[List[str]],
        output_config: BulletinOutputConfig,
        output_dir: str,
        lookback_hours: int
    ) -> None:
        """Internal monitoring loop"""
        # Initial query window
        end = datetime.now(timezone.utc)
        begin = end - timedelta(hours=lookback_hours)

        while not self._stop_event.is_set():
            try:
                if self._status_callback:
                    self._status_callback("Checking for new events...")

                # Query events
                event_ids = self.processor.query_events(begin, end, event_types)

                # Process new events
                new_events = []
                with self._lock:
                    for event_id in event_ids:
                        if event_id not in self.processed_events:
                            new_events.append(event_id)
                            self.processed_events.add(event_id)

                if new_events:
                    self.logger.info(f"Found {len(new_events)} new events")
                    if self._status_callback:
                        self._status_callback(f"Processing {len(new_events)} new events...")

                    for event_id in new_events:
                        if self._stop_event.is_set():
                            break

                        result = self.processor.generator.generate_bulletin(
                            event_id, output_config, output_dir
                        )

                        if self._event_callback:
                            self._event_callback(result)

                # Update time window for next iteration
                begin = end
                end = datetime.now(timezone.utc)

                if self._status_callback:
                    self._status_callback(f"Waiting {self.check_interval}s...")

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                if self._status_callback:
                    self._status_callback(f"Error: {e}")

            # Wait for next check
            self._stop_event.wait(self.check_interval)


# =============================================================================
# CLI Interface
# =============================================================================

def create_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser"""
    parser = argparse.ArgumentParser(
        prog="scbulletin_generator",
        description="SeisComP Bulletin Generator - Query events and generate bulletins in multiple formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Batch mode - query last 24 hours, earthquakes only
  python scbulletin_generator.py batch --hours 24 --event-type earthquake

  # Batch mode - specific time range
  python scbulletin_generator.py batch --begin 2024-01-01T00:00:00 --end 2024-01-02T00:00:00

  # Batch mode - multiple event types
  python scbulletin_generator.py batch --hours 24 -t earthquake -t "quarry blast"

  # Real-time monitoring
  python scbulletin_generator.py realtime --event-type earthquake --interval 60

  # Single event
  python scbulletin_generator.py single --event-id smi:org/event/12345

  # Launch GUI
  python scbulletin_generator.py gui

  # List available event types
  python scbulletin_generator.py list-types

Supported event types:
  earthquake, explosion, quarry blast, chemical explosion, nuclear explosion,
  landslide, debris avalanche, rockslide, mine collapse, volcanic eruption,
  meteor impact, plane crash, building collapse, sonic boom, other
        """
    )

    parser.add_argument(
        "--config", "-c",
        default="config.ini",
        help="Configuration file path (default: config.ini)"
    )

    parser.add_argument(
        "--output-dir", "-o",
        default="output",
        help="Output directory (default: output)"
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )

    parser.add_argument(
        "--version", "-v",
        action="version",
        version=f"%(prog)s {__version__}"
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- BATCH command ---
    batch_parser = subparsers.add_parser("batch", help="Batch processing mode")
    batch_time_group = batch_parser.add_mutually_exclusive_group(required=True)
    batch_time_group.add_argument(
        "--hours",
        type=int,
        help="Query events from last N hours"
    )
    batch_time_group.add_argument(
        "--begin",
        type=str,
        help="Start time (ISO format: YYYY-MM-DDTHH:MM:SS)"
    )
    batch_parser.add_argument(
        "--end",
        type=str,
        help="End time (ISO format, default: now)"
    )
    batch_parser.add_argument(
        "--event-type", "-t",
        action="append",
        dest="event_types",
        metavar="TYPE",
        help="Event type filter (can be specified multiple times)"
    )

    # Output format flags
    batch_parser.add_argument("--no-autoloc3", action="store_true", help="Skip autoloc3 format")
    batch_parser.add_argument("--no-fdsnws", action="store_true", help="Skip FDSNWS format")
    batch_parser.add_argument("--no-kml", action="store_true", help="Skip KML format")
    batch_parser.add_argument("--no-map", action="store_true", help="Skip map image")
    batch_parser.add_argument("--keep-xml", action="store_true", help="Keep intermediate XML files")
    batch_parser.add_argument("--combined", action="store_true", help="Write all autoloc3 bulletins into a single combined text file")

    # --- REALTIME command ---
    realtime_parser = subparsers.add_parser("realtime", help="Real-time monitoring mode")
    realtime_parser.add_argument(
        "--event-type", "-t",
        action="append",
        dest="event_types",
        metavar="TYPE",
        help="Event type filter (can be specified multiple times)"
    )
    realtime_parser.add_argument(
        "--interval", "-i",
        type=int,
        default=60,
        help="Check interval in seconds (default: 60)"
    )
    realtime_parser.add_argument(
        "--lookback",
        type=int,
        default=1,
        help="Initial lookback in hours (default: 1)"
    )
    # Same output format flags as batch
    realtime_parser.add_argument("--no-autoloc3", action="store_true", help="Skip autoloc3 format")
    realtime_parser.add_argument("--no-fdsnws", action="store_true", help="Skip FDSNWS format")
    realtime_parser.add_argument("--no-kml", action="store_true", help="Skip KML format")
    realtime_parser.add_argument("--no-map", action="store_true", help="Skip map image")
    realtime_parser.add_argument("--keep-xml", action="store_true", help="Keep intermediate XML files")

    # --- SINGLE command ---
    single_parser = subparsers.add_parser("single", help="Process single event")
    single_parser.add_argument(
        "--event-id", "-e",
        required=True,
        help="Event ID to process"
    )
    # Same output format flags
    single_parser.add_argument("--no-autoloc3", action="store_true", help="Skip autoloc3 format")
    single_parser.add_argument("--no-fdsnws", action="store_true", help="Skip FDSNWS format")
    single_parser.add_argument("--no-kml", action="store_true", help="Skip KML format")
    single_parser.add_argument("--no-map", action="store_true", help="Skip map image")
    single_parser.add_argument("--keep-xml", action="store_true", help="Keep intermediate XML files")

    # --- GUI command ---
    subparsers.add_parser("gui", help="Launch graphical interface")

    # --- LIST-TYPES command ---
    subparsers.add_parser("list-types", help="List supported event types")

    return parser


def get_output_config(args) -> BulletinOutputConfig:
    """Build BulletinOutputConfig from CLI arguments"""
    return BulletinOutputConfig(
        autoloc3=not getattr(args, 'no_autoloc3', False),
        fdsnws=not getattr(args, 'no_fdsnws', False),
        kml=not getattr(args, 'no_kml', False),
        map_image=not getattr(args, 'no_map', False),
        keep_xml=getattr(args, 'keep_xml', False),
        combined=getattr(args, 'combined', False),
    )


def cli_progress_callback(current: int, total: int, event_id: str) -> None:
    """Progress callback for CLI output"""
    print(f"[{current}/{total}] Processing: {event_id}")


def handle_batch_command(args, config: ConfigManager, logger: logging.Logger) -> int:
    """Handle the batch command"""
    # Parse time range
    if args.hours:
        end = datetime.now(timezone.utc)
        begin = end - timedelta(hours=args.hours)
    else:
        begin = datetime.fromisoformat(args.begin.replace('Z', '+00:00'))
        if not begin.tzinfo:
            begin = begin.replace(tzinfo=timezone.utc)

        if args.end:
            end = datetime.fromisoformat(args.end.replace('Z', '+00:00'))
            if not end.tzinfo:
                end = end.replace(tzinfo=timezone.utc)
        else:
            end = datetime.now(timezone.utc)

    output_config = get_output_config(args)
    processor = BatchProcessor(config, logger)

    print(f"Querying events from {begin} to {end}")
    if args.event_types:
        print(f"Event types: {', '.join(args.event_types)}")

    results = processor.process_time_range(
        begin=begin,
        end=end,
        event_types=args.event_types,
        output_config=output_config,
        output_dir=args.output_dir,
        progress_callback=cli_progress_callback
    )

    # Generate combined autoloc3 file if requested
    if output_config.combined:
        separator = "\n" + "=" * 80 + "\n"
        texts = [r.autoloc3_text for r in results if r.autoloc3_text]
        if texts:
            combined_path = Path(args.output_dir) / "bulletin_combined.txt"
            with open(combined_path, 'w') as f:
                f.write(separator.join(texts))
            print(f"\nCombined autoloc3 bulletin: {combined_path}")
        else:
            print("\nNo autoloc3 bulletins to combine")

    # Summary
    success_count = sum(1 for r in results if r.is_success())
    error_count = sum(1 for r in results if r.error)

    print(f"\nCompleted: {success_count} successful, {error_count} errors")
    print(f"Output saved to: {args.output_dir}")

    return 0 if error_count == 0 else 1


def handle_realtime_command(args, config: ConfigManager, logger: logging.Logger) -> int:
    """Handle the realtime command"""
    output_config = get_output_config(args)

    def event_callback(result: BulletinOutput):
        status = "OK" if result.is_success() else f"ERROR: {result.error}"
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {result.event_id}: {status}")

    def status_callback(message: str):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

    monitor = RealTimeMonitor(config, logger, check_interval=args.interval)

    print(f"Starting real-time monitoring (interval: {args.interval}s, lookback: {args.lookback}h)")
    if args.event_types:
        print(f"Event types: {', '.join(args.event_types)}")
    print("Press Ctrl+C to stop\n")

    monitor.start(
        event_types=args.event_types,
        output_config=output_config,
        output_dir=args.output_dir,
        lookback_hours=args.lookback,
        event_callback=event_callback,
        status_callback=status_callback
    )

    try:
        while monitor.is_running:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
        monitor.stop()

    status = monitor.get_status()
    print(f"\nProcessed {status['processed_count']} events")
    return 0


def handle_single_command(args, config: ConfigManager, logger: logging.Logger) -> int:
    """Handle the single command"""
    output_config = get_output_config(args)
    generator = BulletinGenerator(config, logger)

    print(f"Processing event: {args.event_id}")

    result = generator.generate_bulletin(
        args.event_id,
        output_config,
        args.output_dir
    )

    if result.is_success():
        print(f"\nSuccess! Processing time: {result.processing_time:.2f}s")
        if result.autoloc3_file:
            print(f"  Autoloc3: {result.autoloc3_file}")
        if result.fdsnws_file:
            print(f"  FDSNWS: {result.fdsnws_file}")
        if result.kml_file:
            print(f"  KML: {result.kml_file}")
        if result.map_file:
            print(f"  Map: {result.map_file}")
        if result.xml_file:
            print(f"  XML: {result.xml_file}")
        return 0
    else:
        print(f"\nError: {result.error}")
        return 1


def handle_list_types_command() -> int:
    """Handle the list-types command"""
    print("Supported SeisComP event types:\n")
    for event_type in EVENT_TYPES:
        print(f"  - {event_type}")
    return 0


def handle_gui_command(args, config: ConfigManager, logger: logging.Logger) -> int:
    """Handle the gui command"""
    try:
        from scbulletin_generator_gui import launch_gui
        return launch_gui(config, logger)
    except ImportError:
        # GUI code is embedded below
        pass

    # Check for PyQt5
    try:
        from PyQt5.QtWidgets import QApplication
        from PyQt5.QtCore import Qt
    except ImportError:
        print("Error: PyQt5 is required for the GUI.")
        print("Install it with: pip install PyQt5")
        return 1

    # Launch embedded GUI
    return launch_embedded_gui(args, config, logger)


# =============================================================================
# Embedded PyQt5 GUI
# =============================================================================

def launch_embedded_gui(args, config: ConfigManager, logger: logging.Logger) -> int:
    """Launch the embedded PyQt5 GUI"""
    try:
        from PyQt5.QtWidgets import (
            QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
            QTabWidget, QGroupBox, QLabel, QLineEdit, QPushButton, QCheckBox,
            QSpinBox, QDoubleSpinBox, QTableWidget, QTableWidgetItem, QTextEdit,
            QSplitter, QFileDialog, QMessageBox, QProgressBar, QStatusBar,
            QDateTimeEdit, QScrollArea, QGridLayout, QHeaderView, QComboBox,
            QAction
        )
        from PyQt5.QtCore import Qt, QThread, pyqtSignal, QDateTime
        from PyQt5.QtGui import QPixmap, QFont
    except ImportError:
        print("Error: PyQt5 is required for the GUI.")
        print("Install it with: pip install PyQt5")
        return 1

    class WorkerThread(QThread):
        """Background worker thread for processing"""
        progress = pyqtSignal(int, int, str)
        result = pyqtSignal(object)
        error = pyqtSignal(str)
        finished_signal = pyqtSignal()

        def __init__(self, task_func, *args, **kwargs):
            super().__init__()
            self.task_func = task_func
            self.task_args = args
            self.task_kwargs = kwargs

        def run(self):
            try:
                result = self.task_func(*self.task_args, **self.task_kwargs)
                self.result.emit(result)
            except Exception as e:
                self.error.emit(str(e))
            finally:
                self.finished_signal.emit()

    class EventTypeSelector(QGroupBox):
        """Widget for selecting event types"""

        def __init__(self, parent=None):
            super().__init__("Event Types", parent)
            self._setup_ui()

        def _setup_ui(self):
            layout = QGridLayout()

            self.checkboxes = {}
            for i, event_type in enumerate(EVENT_TYPES):
                cb = QCheckBox(event_type)
                self.checkboxes[event_type] = cb
                row, col = divmod(i, 3)
                layout.addWidget(cb, row, col)

            # Buttons
            btn_layout = QHBoxLayout()
            select_all_btn = QPushButton("Select All")
            select_all_btn.clicked.connect(self._select_all)
            clear_btn = QPushButton("Clear")
            clear_btn.clicked.connect(self._clear_all)
            btn_layout.addWidget(select_all_btn)
            btn_layout.addWidget(clear_btn)
            btn_layout.addStretch()

            main_layout = QVBoxLayout()
            main_layout.addLayout(layout)
            main_layout.addLayout(btn_layout)
            self.setLayout(main_layout)

        def _select_all(self):
            for cb in self.checkboxes.values():
                cb.setChecked(True)

        def _clear_all(self):
            for cb in self.checkboxes.values():
                cb.setChecked(False)

        def get_selected_types(self) -> List[str]:
            return [t for t, cb in self.checkboxes.items() if cb.isChecked()]

    class OutputFormatSelector(QGroupBox):
        """Widget for selecting output formats"""

        def __init__(self, parent=None):
            super().__init__("Output Formats", parent)
            self._setup_ui()

        def _setup_ui(self):
            layout = QHBoxLayout()

            self.autoloc3_cb = QCheckBox("Text Bulletin (-3 -k -e)")
            self.autoloc3_cb.setChecked(True)
            self.fdsnws_cb = QCheckBox("FDSNWS (-4)")
            self.fdsnws_cb.setChecked(True)
            self.kml_cb = QCheckBox("KML")
            self.kml_cb.setChecked(True)
            self.map_cb = QCheckBox("Map Image")
            self.map_cb.setChecked(True)
            self.keep_xml_cb = QCheckBox("Keep XML")
            self.keep_xml_cb.setChecked(False)
            self.combined_cb = QCheckBox("Combined File")
            self.combined_cb.setChecked(False)

            layout.addWidget(self.autoloc3_cb)
            layout.addWidget(self.fdsnws_cb)
            layout.addWidget(self.kml_cb)
            layout.addWidget(self.map_cb)
            layout.addWidget(self.keep_xml_cb)
            layout.addWidget(self.combined_cb)
            layout.addStretch()

            self.setLayout(layout)

        def get_config(self) -> BulletinOutputConfig:
            return BulletinOutputConfig(
                autoloc3=self.autoloc3_cb.isChecked(),
                fdsnws=self.fdsnws_cb.isChecked(),
                kml=self.kml_cb.isChecked(),
                map_image=self.map_cb.isChecked(),
                keep_xml=self.keep_xml_cb.isChecked(),
                combined=self.combined_cb.isChecked(),
            )

    class MainWindow(QMainWindow):
        """Main application window"""

        def __init__(self, config: ConfigManager, logger: logging.Logger):
            super().__init__()
            self.config = config
            self.logger = logger
            self.processor = BatchProcessor(config, logger)
            self.monitor = RealTimeMonitor(config, logger)
            self.event_results: Dict[str, BulletinOutput] = {}
            self.worker: Optional[WorkerThread] = None

            self._setup_ui()
            self._connect_signals()

        def _setup_ui(self):
            self.setWindowTitle(f"SeisComP Bulletin Generator v{__version__}")
            self.setMinimumSize(1200, 800)

            # Central widget
            central = QWidget()
            self.setCentralWidget(central)
            main_layout = QVBoxLayout(central)

            # Mode tabs
            self.mode_tabs = QTabWidget()

            # Batch mode tab
            batch_widget = self._create_batch_tab()
            self.mode_tabs.addTab(batch_widget, "Batch Mode")

            # Real-time mode tab
            realtime_widget = self._create_realtime_tab()
            self.mode_tabs.addTab(realtime_widget, "Real-Time Monitor")

            main_layout.addWidget(self.mode_tabs)

            # Output format selector
            self.output_format_selector = OutputFormatSelector()
            main_layout.addWidget(self.output_format_selector)

            # Splitter for event list and output viewer
            splitter = QSplitter(Qt.Horizontal)

            # Event list
            event_list_widget = self._create_event_list()
            splitter.addWidget(event_list_widget)

            # Output viewer
            output_viewer = self._create_output_viewer()
            splitter.addWidget(output_viewer)

            splitter.setSizes([400, 600])
            main_layout.addWidget(splitter, stretch=1)

            # Status bar
            self.status_bar = QStatusBar()
            self.setStatusBar(self.status_bar)
            self.progress_bar = QProgressBar()
            self.progress_bar.setMaximumWidth(200)
            self.progress_bar.hide()
            self.status_bar.addPermanentWidget(self.progress_bar)
            self.status_bar.showMessage("Ready")

            # Menu bar
            self._create_menu_bar()

        def _create_batch_tab(self) -> QWidget:
            widget = QWidget()
            layout = QVBoxLayout(widget)

            # Time range
            time_group = QGroupBox("Time Range")
            time_layout = QHBoxLayout()

            time_layout.addWidget(QLabel("Start:"))
            self.start_datetime = QDateTimeEdit()
            self.start_datetime.setCalendarPopup(True)
            self.start_datetime.setDateTime(
                QDateTime.currentDateTime().addDays(-1)
            )
            time_layout.addWidget(self.start_datetime)

            time_layout.addWidget(QLabel("End:"))
            self.end_datetime = QDateTimeEdit()
            self.end_datetime.setCalendarPopup(True)
            self.end_datetime.setDateTime(QDateTime.currentDateTime())
            time_layout.addWidget(self.end_datetime)

            # Quick presets
            self.hours_spinbox = QSpinBox()
            self.hours_spinbox.setRange(1, 720)
            self.hours_spinbox.setValue(24)
            last_hours_btn = QPushButton("Last N hours")
            last_hours_btn.clicked.connect(self._set_last_hours)
            time_layout.addWidget(self.hours_spinbox)
            time_layout.addWidget(last_hours_btn)

            time_layout.addStretch()
            time_group.setLayout(time_layout)
            layout.addWidget(time_group)

            # Event type selector
            self.batch_event_types = EventTypeSelector()
            layout.addWidget(self.batch_event_types)

            # Query button
            btn_layout = QHBoxLayout()
            self.query_btn = QPushButton("Query Events")
            self.query_btn.clicked.connect(self._on_query_events)
            self.generate_all_btn = QPushButton("Generate All Bulletins")
            self.generate_all_btn.clicked.connect(self._on_generate_all)
            self.generate_all_btn.setEnabled(False)
            btn_layout.addWidget(self.query_btn)
            btn_layout.addWidget(self.generate_all_btn)
            btn_layout.addStretch()
            layout.addLayout(btn_layout)

            layout.addStretch()
            return widget

        def _create_realtime_tab(self) -> QWidget:
            widget = QWidget()
            layout = QVBoxLayout(widget)

            # Settings
            settings_group = QGroupBox("Monitor Settings")
            settings_layout = QHBoxLayout()

            settings_layout.addWidget(QLabel("Check Interval (s):"))
            self.interval_spinbox = QSpinBox()
            self.interval_spinbox.setRange(10, 3600)
            self.interval_spinbox.setValue(60)
            settings_layout.addWidget(self.interval_spinbox)

            settings_layout.addWidget(QLabel("Lookback (hours):"))
            self.lookback_spinbox = QSpinBox()
            self.lookback_spinbox.setRange(1, 48)
            self.lookback_spinbox.setValue(1)
            settings_layout.addWidget(self.lookback_spinbox)

            settings_layout.addStretch()
            settings_group.setLayout(settings_layout)
            layout.addWidget(settings_group)

            # Event type selector
            self.realtime_event_types = EventTypeSelector()
            layout.addWidget(self.realtime_event_types)

            # Control buttons
            btn_layout = QHBoxLayout()
            self.start_monitor_btn = QPushButton("Start Monitoring")
            self.start_monitor_btn.clicked.connect(self._on_start_monitoring)
            self.stop_monitor_btn = QPushButton("Stop Monitoring")
            self.stop_monitor_btn.clicked.connect(self._on_stop_monitoring)
            self.stop_monitor_btn.setEnabled(False)

            self.monitor_status_label = QLabel("Status: Stopped")

            btn_layout.addWidget(self.start_monitor_btn)
            btn_layout.addWidget(self.stop_monitor_btn)
            btn_layout.addWidget(self.monitor_status_label)
            btn_layout.addStretch()
            layout.addLayout(btn_layout)

            layout.addStretch()
            return widget

        def _create_event_list(self) -> QWidget:
            widget = QGroupBox("Events")
            layout = QVBoxLayout(widget)

            self.event_table = QTableWidget()
            self.event_table.setColumnCount(3)
            self.event_table.setHorizontalHeaderLabels(["Event ID", "Status", "Time"])
            self.event_table.horizontalHeader().setSectionResizeMode(
                0, QHeaderView.Stretch
            )
            self.event_table.setSelectionBehavior(
                QTableWidget.SelectRows
            )
            self.event_table.itemSelectionChanged.connect(self._on_event_selected)
            layout.addWidget(self.event_table)

            # Buttons
            btn_layout = QHBoxLayout()
            self.generate_selected_btn = QPushButton("Generate Selected")
            self.generate_selected_btn.clicked.connect(self._on_generate_selected)
            self.generate_selected_btn.setEnabled(False)
            self.clear_btn = QPushButton("Clear")
            self.clear_btn.clicked.connect(self._on_clear_events)
            btn_layout.addWidget(self.generate_selected_btn)
            btn_layout.addWidget(self.clear_btn)
            btn_layout.addStretch()
            layout.addLayout(btn_layout)

            return widget

        def _create_output_viewer(self) -> QWidget:
            widget = QGroupBox("Output Preview")
            layout = QVBoxLayout(widget)

            self.output_tabs = QTabWidget()

            # Text views
            self.autoloc3_view = QTextEdit()
            self.autoloc3_view.setReadOnly(True)
            self.autoloc3_view.setFont(QFont("Monospace", 10))
            self.output_tabs.addTab(self.autoloc3_view, "Autoloc3")

            self.fdsnws_view = QTextEdit()
            self.fdsnws_view.setReadOnly(True)
            self.fdsnws_view.setFont(QFont("Monospace", 10))
            self.output_tabs.addTab(self.fdsnws_view, "FDSNWS")

            # Map view
            self.map_label = QLabel()
            self.map_label.setAlignment(Qt.AlignCenter)
            scroll = QScrollArea()
            scroll.setWidget(self.map_label)
            scroll.setWidgetResizable(True)
            self.output_tabs.addTab(scroll, "Map")

            layout.addWidget(self.output_tabs)

            # Save buttons
            btn_layout = QHBoxLayout()
            self.open_dir_btn = QPushButton("Open Output Directory")
            self.open_dir_btn.clicked.connect(self._on_open_output_dir)
            btn_layout.addWidget(self.open_dir_btn)
            btn_layout.addStretch()
            layout.addLayout(btn_layout)

            return widget

        def _create_menu_bar(self):
            menubar = self.menuBar()

            # File menu
            file_menu = menubar.addMenu("File")

            open_config_action = QAction("Open Config...", self)
            open_config_action.triggered.connect(self._on_open_config)
            file_menu.addAction(open_config_action)

            file_menu.addSeparator()

            exit_action = QAction("Exit", self)
            exit_action.triggered.connect(self.close)
            file_menu.addAction(exit_action)

            # Help menu
            help_menu = menubar.addMenu("Help")

            about_action = QAction("About", self)
            about_action.triggered.connect(self._on_about)
            help_menu.addAction(about_action)

        def _connect_signals(self):
            pass

        def _set_last_hours(self):
            hours = self.hours_spinbox.value()
            self.end_datetime.setDateTime(QDateTime.currentDateTime())
            self.start_datetime.setDateTime(
                QDateTime.currentDateTime().addSecs(-hours * 3600)
            )

        def _on_query_events(self):
            begin = self.start_datetime.dateTime().toPyDateTime()
            end = self.end_datetime.dateTime().toPyDateTime()
            event_types = self.batch_event_types.get_selected_types() or None

            self.status_bar.showMessage("Querying events...")
            self.query_btn.setEnabled(False)

            def query_task():
                return self.processor.query_events(begin, end, event_types)

            self.worker = WorkerThread(query_task)
            self.worker.result.connect(self._on_query_complete)
            self.worker.error.connect(self._on_query_error)
            self.worker.finished_signal.connect(lambda: self.query_btn.setEnabled(True))
            self.worker.start()

        def _on_query_complete(self, event_ids: List[str]):
            self.event_table.setRowCount(0)
            self.event_results.clear()

            for event_id in event_ids:
                row = self.event_table.rowCount()
                self.event_table.insertRow(row)
                self.event_table.setItem(row, 0, QTableWidgetItem(event_id))
                self.event_table.setItem(row, 1, QTableWidgetItem("Pending"))
                self.event_table.setItem(row, 2, QTableWidgetItem(""))

            self.generate_all_btn.setEnabled(len(event_ids) > 0)
            self.generate_selected_btn.setEnabled(len(event_ids) > 0)
            self.status_bar.showMessage(f"Found {len(event_ids)} events")

        def _on_query_error(self, error: str):
            QMessageBox.critical(self, "Error", f"Query failed: {error}")
            self.status_bar.showMessage("Query failed")

        def _on_generate_all(self):
            if self.event_table.rowCount() == 0:
                return

            event_ids = []
            for row in range(self.event_table.rowCount()):
                event_id = self.event_table.item(row, 0).text()
                event_ids.append(event_id)

            self._generate_bulletins(event_ids)

        def _on_generate_selected(self):
            selected_rows = set(item.row() for item in self.event_table.selectedItems())
            if not selected_rows:
                return

            event_ids = []
            for row in selected_rows:
                event_id = self.event_table.item(row, 0).text()
                event_ids.append(event_id)

            self._generate_bulletins(event_ids)

        def _generate_bulletins(self, event_ids: List[str]):
            output_config = self.output_format_selector.get_config()
            output_dir = self.config.get_output_directory()

            self.progress_bar.setMaximum(len(event_ids))
            self.progress_bar.setValue(0)
            self.progress_bar.show()

            self.query_btn.setEnabled(False)
            self.generate_all_btn.setEnabled(False)
            self.generate_selected_btn.setEnabled(False)

            def generate_task():
                results = []
                for i, event_id in enumerate(event_ids, 1):
                    result = self.processor.generator.generate_bulletin(
                        event_id, output_config, output_dir
                    )
                    results.append(result)
                    # Update progress in main thread
                    self.progress_bar.setValue(i)
                return results

            self.worker = WorkerThread(generate_task)
            self.worker.result.connect(self._on_generate_complete)
            self.worker.error.connect(self._on_generate_error)
            self.worker.finished_signal.connect(self._on_generate_finished)
            self.worker.start()

        def _on_generate_complete(self, results: List[BulletinOutput]):
            for result in results:
                self.event_results[result.event_id] = result

                # Update table
                for row in range(self.event_table.rowCount()):
                    if self.event_table.item(row, 0).text() == result.event_id:
                        status = "Complete" if result.is_success() else "Error"
                        self.event_table.setItem(row, 1, QTableWidgetItem(status))
                        break

            # Generate combined autoloc3 file if requested
            output_config = self.output_format_selector.get_config()
            if output_config.combined:
                separator = "\n" + "=" * 80 + "\n"
                texts = [r.autoloc3_text for r in results if r.autoloc3_text]
                if texts:
                    output_dir = self.config.get_output_directory()
                    combined_path = Path(output_dir) / "bulletin_combined.txt"
                    with open(combined_path, 'w') as f:
                        f.write(separator.join(texts))
                    QMessageBox.information(
                        self, "Combined Bulletin",
                        f"Combined autoloc3 bulletin saved to:\n{combined_path}"
                    )

            success_count = sum(1 for r in results if r.is_success())
            self.status_bar.showMessage(
                f"Generated {success_count}/{len(results)} bulletins"
            )

        def _on_generate_error(self, error: str):
            QMessageBox.critical(self, "Error", f"Generation failed: {error}")

        def _on_generate_finished(self):
            self.progress_bar.hide()
            self.query_btn.setEnabled(True)
            self.generate_all_btn.setEnabled(True)
            self.generate_selected_btn.setEnabled(True)

        def _on_event_selected(self):
            selected = self.event_table.selectedItems()
            if not selected:
                return

            row = selected[0].row()
            event_id = self.event_table.item(row, 0).text()

            if event_id in self.event_results:
                result = self.event_results[event_id]
                self._display_result(result)

        def _display_result(self, result: BulletinOutput):
            # Autoloc3
            if result.autoloc3_text:
                self.autoloc3_view.setText(result.autoloc3_text)
            else:
                self.autoloc3_view.setText("Not available")

            # FDSNWS
            if result.fdsnws_text:
                self.fdsnws_view.setText(result.fdsnws_text)
            else:
                self.fdsnws_view.setText("Not available")

            # Map
            if result.map_file and Path(result.map_file).exists():
                pixmap = QPixmap(result.map_file)
                self.map_label.setPixmap(
                    pixmap.scaled(800, 600, Qt.KeepAspectRatio)
                )
            else:
                self.map_label.setText("Map not available")

        def _on_clear_events(self):
            self.event_table.setRowCount(0)
            self.event_results.clear()
            self.autoloc3_view.clear()
            self.fdsnws_view.clear()
            self.map_label.clear()
            self.generate_all_btn.setEnabled(False)
            self.generate_selected_btn.setEnabled(False)

        def _on_start_monitoring(self):
            output_config = self.output_format_selector.get_config()
            output_dir = self.config.get_output_directory()
            event_types = self.realtime_event_types.get_selected_types() or None
            interval = self.interval_spinbox.value()
            lookback = self.lookback_spinbox.value()

            def event_callback(result: BulletinOutput):
                self.event_results[result.event_id] = result
                # Add to table
                row = self.event_table.rowCount()
                self.event_table.insertRow(row)
                self.event_table.setItem(row, 0, QTableWidgetItem(result.event_id))
                status = "Complete" if result.is_success() else "Error"
                self.event_table.setItem(row, 1, QTableWidgetItem(status))
                self.event_table.setItem(
                    row, 2, QTableWidgetItem(datetime.now().strftime("%H:%M:%S"))
                )

            def status_callback(message: str):
                self.monitor_status_label.setText(f"Status: {message}")

            self.monitor.start(
                event_types=event_types,
                output_config=output_config,
                output_dir=output_dir,
                lookback_hours=lookback,
                event_callback=event_callback,
                status_callback=status_callback
            )

            self.start_monitor_btn.setEnabled(False)
            self.stop_monitor_btn.setEnabled(True)
            self.monitor_status_label.setText("Status: Running")

        def _on_stop_monitoring(self):
            self.monitor.stop()
            self.start_monitor_btn.setEnabled(True)
            self.stop_monitor_btn.setEnabled(False)
            self.monitor_status_label.setText("Status: Stopped")

        def _on_open_output_dir(self):
            output_dir = self.config.get_output_directory()
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            import subprocess
            import platform

            if platform.system() == "Windows":
                os.startfile(output_dir)
            elif platform.system() == "Darwin":
                subprocess.run(["open", output_dir])
            else:
                subprocess.run(["xdg-open", output_dir])

        def _on_open_config(self):
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Open Configuration", "", "INI files (*.ini);;All files (*)"
            )
            if file_path:
                try:
                    self.config = ConfigManager(file_path)
                    self.config.load()
                    self.processor = BatchProcessor(self.config, self.logger)
                    self.monitor = RealTimeMonitor(self.config, self.logger)
                    self.status_bar.showMessage(f"Loaded config: {file_path}")
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to load config: {e}")

        def _on_about(self):
            QMessageBox.about(
                self,
                "About SeisComP Bulletin Generator",
                f"SeisComP Bulletin Generator v{__version__}\n\n"
                f"Author: {__author__}\n\n"
                "A tool for querying seismic events and generating bulletins "
                "in multiple formats using SeisComP native tools."
            )

        def closeEvent(self, event):
            if self.monitor.is_running:
                self.monitor.stop()
            event.accept()

    # Launch the application
    app = QApplication(sys.argv)
    window = MainWindow(config, logger)
    window.show()
    return app.exec()


# =============================================================================
# Main Entry Point
# =============================================================================

def main() -> int:
    """Main entry point"""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Setup logging
    logger = ProductionLogger.setup_logging(
        log_file="logs/bulletin_generator.log",
        log_level=args.log_level
    )

    # Handle list-types command (no config needed)
    if args.command == "list-types":
        return handle_list_types_command()

    # Load configuration
    try:
        config = ConfigManager(args.config)
        config.load()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Create a configuration file or specify one with --config")
        return 1

    # Handle commands
    if args.command == "batch":
        return handle_batch_command(args, config, logger)
    elif args.command == "realtime":
        return handle_realtime_command(args, config, logger)
    elif args.command == "single":
        return handle_single_command(args, config, logger)
    elif args.command == "gui":
        return handle_gui_command(args, config, logger)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
