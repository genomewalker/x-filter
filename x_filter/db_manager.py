import os
import logging
import duckdb
import psutil
from pathlib import Path
import tempfile
import time
import json
import re
import io
import sys
from contextlib import redirect_stdout
from typing import Optional, Union
from x_filter.logging_setup import get_logger, is_debug

log = get_logger()


def parse_memory_string(memory_str: str) -> int:
    """
    Parse memory string (e.g., '250G', '16GB', '2048M') into bytes.

    Args:
        memory_str: Memory specification string

    Returns:
        Memory size in bytes
    """
    if not isinstance(memory_str, str):
        return int(memory_str)  # Already in bytes

    memory_str = memory_str.strip().upper()

    # Handle different unit formats
    multipliers = {
        'B': 1,
        'K': 1024, 'KB': 1024,
        'M': 1024**2, 'MB': 1024**2,
        'G': 1024**3, 'GB': 1024**3,
        'T': 1024**4, 'TB': 1024**4
    }

    # Extract number and unit
    import re
    match = re.match(r'^(\d+(?:\.\d+)?)\s*([KMGT]B?)?\s*$', memory_str)
    if not match:
        raise ValueError(f"Invalid memory format: {memory_str}")

    number = float(match.group(1))
    unit = match.group(2) or 'B'

    if unit not in multipliers:
        raise ValueError(f"Unknown memory unit: {unit}")

    return int(number * multipliers[unit])


class DatabaseManager:
    """Manager for DuckDB connections with standardized configuration and utility methods."""

    def __init__(
        self,
        database: Optional[str] = None,
        temp_dir: Optional[str] = None,
        threads: Optional[int] = None,
        memory_limit: Optional[Union[str, int]] = None,
        max_memory_pct: int = 60,
        enable_progress: bool = False,
        checkpoint_threshold: str = "1GB",
        profile_output_dir: Optional[str] = None,  # Added missing parameter
    ):
        """
        Initialize a new DatabaseManager.

        Args:
            database (str): Path to persistent database or ":memory:" for in-memory database
            temp_dir (str): Directory to store temporary files (None = use system temp)
            threads (int): Number of threads to use (None = auto-detect)
            memory_limit (str): Memory limit with units like "4GB" (None = auto-detect)
            max_memory_pct (int): Percentage of system memory to use when auto-detecting
            enable_progress (bool): Enable progress bar
            profile_output_dir (str): Directory to save profiling output (None = auto-create if debug mode)
        """
        # Handle temporary directory first
        if temp_dir is None:
            self.temp_dir = tempfile.gettempdir()
        else:
            self.temp_dir = temp_dir

        # Create the temp directory if it doesn't exist
        if not os.path.exists(self.temp_dir):
            try:
                os.makedirs(self.temp_dir, exist_ok=True)
                log.info(f"Created temporary directory: {self.temp_dir}")
            except Exception as e:
                log.warning(f"Could not create temp directory {self.temp_dir}: {e}")
                self.temp_dir = tempfile.gettempdir()
                log.info(f"Using system temp directory instead: {self.temp_dir}")

        # Check write permissions on temp directory
        if not os.access(self.temp_dir, os.W_OK):
            log.warning(f"No write permission on {self.temp_dir}")
            self.temp_dir = tempfile.gettempdir()
            log.info(f"Using system temp directory instead: {self.temp_dir}")

        # Always use a file-based database (never pure in‐memory)
        if database is None or database == ":memory:":
            try:
                tmp_db = tempfile.NamedTemporaryFile(
                    delete=False, suffix=".db", dir=self.temp_dir
                )
                tmp_db_path = tmp_db.name
                tmp_db.close()  # Close the file handle immediately

                # Ensure the file does not exist before DuckDB tries to use it
                if os.path.exists(tmp_db_path):
                    log.debug(f"Deleting potentially pre-existing temp db file: {tmp_db_path}")
                    os.unlink(tmp_db_path)

                self.database = tmp_db_path
                self._temp_db_created = True

            except Exception as e:
                log.error(f"Failed to create or manage temporary database file: {e}")
                raise

            if database == ":memory:":
                log.info(f"Using temporary file-based database instead of in-memory: {self.database}")
            else:
                log.info(f"No database specified, created temporary file: {self.database}")
        else:
            self.database = database
            self._temp_db_created = False
            log.info(f"Using specified database file: {self.database}")

        self.threads = threads if threads else max(4, os.cpu_count())

        # Parse memory limit properly
        if memory_limit is not None:
            if isinstance(memory_limit, str):
                self.memory_limit_bytes = parse_memory_string(memory_limit)
                log.info(f"Using explicit memory limit: {memory_limit} ({self.memory_limit_bytes:,} bytes)")
            else:
                self.memory_limit_bytes = int(memory_limit)
                log.info(f"Using explicit memory limit: {self.memory_limit_bytes:,} bytes")
        else:
            # Only auto-configure if no explicit limit provided
            available_memory = psutil.virtual_memory().available
            self.memory_limit_bytes = int(available_memory * (max_memory_pct / 100))
            log.info(f"Auto-configured memory limit to {self.memory_limit_bytes // (1024**3)}GB ({max_memory_pct}% of available memory)")

        # Convert to DuckDB format
        if self.memory_limit_bytes >= 1024**3:  # 1GB or more
            self.duckdb_memory_limit = f"{self.memory_limit_bytes // (1024**3)}GB"
        elif self.memory_limit_bytes >= 1024**2:  # 1MB or more
            self.duckdb_memory_limit = f"{self.memory_limit_bytes // (1024**2)}MB"
        else:
            self.duckdb_memory_limit = f"{self.memory_limit_bytes}B"

        self.enable_progress = enable_progress

        # Set up profiling directory with auto-creation
        if profile_output_dir is None and is_debug():
            # Auto-create profiling directory in temp_dir when in debug mode
            profile_output_dir = os.path.join(self.temp_dir, "profiling")
            log.info(f"Debug mode detected: auto-creating profiling directory at {profile_output_dir}")
        
        self.profile_output_dir = profile_output_dir
        self.profiling_enabled = is_debug() and profile_output_dir is not None

        if self.profiling_enabled:
            try:
                os.makedirs(self.profile_output_dir, exist_ok=True)
                log.info(f"DuckDB profiling enabled, output saved to: {self.profile_output_dir}")
                # Also display this on screen for user visibility
                print(f"📊 DuckDB profiling enabled - logs saved to: {self.profile_output_dir}")
            except Exception as e:
                log.warning(f"Could not create profiling directory {self.profile_output_dir}: {e}")
                self.profiling_enabled = False
        elif is_debug():
            log.info("Debug mode active but no profiling directory specified")

        self.con = None
        self.query_count = 0
        self.supported_pragmas = None

    def __enter__(self):
        """Context manager entry point that connects to the database."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point that closes the database connection."""
        if self.con:
            if self.profiling_enabled:
                try:
                    self._save_profiling_output("final_profiling")
                except Exception as e:
                    log.debug(f"Could not save final profiling data: {e}")
            
            # Properly close the connection first
            try:
                self.con.close()
                self.con = None
                log.debug("DuckDB connection closed successfully")
            except Exception as e:
                log.debug(f"Error closing DuckDB connection: {e}")

        # Remove temporary DB file if we created it
        if getattr(self, "_temp_db_created", False):
            try:
                # Add a small delay to ensure file handles are released
                import time
                time.sleep(0.1)
                
                if os.path.exists(self.database):
                    log.debug(f"Cleaning up temporary database file: {self.database}")
                    # Try multiple times with increasing delays for NFS filesystems
                    for attempt in range(3):
                        try:
                            os.unlink(self.database)
                            log.debug("Temporary database file removed successfully")
                            break
                        except OSError as cleanup_error:
                            if attempt < 2:  # Not the last attempt
                                log.debug(f"Cleanup attempt {attempt + 1} failed, retrying: {cleanup_error}")
                                time.sleep(0.5 * (attempt + 1))  # Increasing delay
                            else:
                                log.warning(f"Could not remove temporary database file after 3 attempts: {cleanup_error}")
            except Exception as e:
                log.warning(f"Could not remove temporary database file {self.database}: {e}")

    @property
    def connection(self):
        """Alias for self.con for backward compatibility."""
        return self.con

    def connect(self):
        """Connect to the DuckDB database and configure it."""
        if self.con:
            log.warning("Connection already established, reusing existing connection")
            return self.con

        log.debug(f"Connecting to DuckDB database: {self.database}")

        try:
            # First attempt with modern config
            self.con = duckdb.connect(database=self.database, config={'external_access': 'true'})
            log.info("Successfully connected with external access enabled via config.")
        except Exception as e:
            try:
                # Fall back to basic connection for older versions
                self.con = duckdb.connect(self.database)
                log.info("Connected using compatibility mode.")
            except Exception as e2:
                log.error(f"Failed to connect to database: {e2}")
                self.con = None  # Ensure connection is None on failure
                raise

        if self.con:  # Only configure if connection was successful
            # Configure the connection
            self._configure_connection()
            # Discover supported pragmas
            self._discover_supported_pragmas()

        return self.con

    def _discover_supported_pragmas(self):
        """Discover which pragmas are supported by this DuckDB version."""
        self.supported_pragmas = set()
        try:
            pragma_result = self.con.execute("PRAGMA pragma_database_list").fetchall()
            for row in pragma_result:
                self.supported_pragmas.add(row[0])
            log.debug(f"Discovered {len(self.supported_pragmas)} supported pragmas")
        except Exception as e:
            log.debug(f"Could not discover supported pragmas: {e}")
            self.supported_pragmas = {"threads", "memory_limit", "enable_profiling", "disable_profiling"}

    def _configure_connection(self):
        """Apply configuration settings to the connection."""
        try:
            # Set critical settings first
            self.con.execute(f"SET threads TO {self.threads}")

            # Handle memory limit with better validation - use the correct attribute
            memory_limit = self._validate_memory_limit(self.duckdb_memory_limit)
            log.info(f"Setting DuckDB memory limit to: {memory_limit}")
            self.con.execute(f"SET memory_limit='{memory_limit}'")

            # Verify the memory limit was set correctly
            try:
                current_limit = self.con.execute("SELECT current_setting('memory_limit')").fetchone()
                if current_limit:
                    log.debug(f"Confirmed memory limit set to: {current_limit[0]}")
            except Exception as e:
                log.debug(f"Could not verify memory limit setting: {e}")

            # Set temp directory - critical for large dataset processing
            if self.temp_dir:
                log.info(f"Setting DuckDB temp directory to: {self.temp_dir}")
                safe_temp_dir = str(Path(self.temp_dir).resolve()).replace("'", "''")
                self.con.execute(f"SET temp_directory='{safe_temp_dir}'")

            # Conservative settings for large dataset processing
            self.con.execute("SET preserve_insertion_order=false")
            self.con.execute("SET immediate_transaction_mode=false")  # More conservative

            # Memory management settings - more conservative for OOM situations
            try:
                self.con.execute("SET checkpoint_threshold='64MB'")  # Even more frequent checkpoints
                log.debug("Set checkpoint_threshold to 64MB")
            except Exception as e:
                log.debug(f"Could not set checkpoint_threshold: {e}")

            try:
                self.con.execute("SET allocator_flush_threshold='32MB'")  # More aggressive flushing
                log.debug("Set allocator_flush_threshold to 32MB")
            except Exception as e:
                log.debug(f"Could not set allocator_flush_threshold: {e}")

            # Streaming settings for large datasets
            try:
                self.con.execute("SET streaming_buffer_size='16MB'")  # Even smaller buffer
                log.debug("Set streaming_buffer_size to 16MB")
            except Exception as e:
                log.debug(f"Could not set streaming_buffer_size: {e}")

            # Remove problematic settings that are causing parser errors
            # Additional memory-conservative settings that work
            try:
                # Use a more conservative buffer pool size
                self.con.execute("SET buffer_manager_size='256MB'")
                log.debug("Set buffer_manager_size to 256MB")
            except Exception as e:
                log.debug(f"Could not set buffer_manager_size: {e}")

            # Enable progress bar if requested
            if self.enable_progress:
                try:
                    self.con.execute("SET progress_bar_time=1000")
                    self.con.execute("SET enable_progress_bar=true")
                    log.info("Enabled progress bar for long-running queries")
                except Exception as e:
                    log.debug(f"Could not enable progress bar: {e}")

            # Profiling setup
            if self.profiling_enabled:
                try:
                    self.con.execute("PRAGMA enable_profiling")
                    log.debug("DuckDB profiling enabled because log level is DEBUG.")
                    try:
                        self.con.execute("PRAGMA profiling_mode='detailed'")
                        log.debug("DuckDB detailed profiling enabled")
                    except:
                        log.debug("Detailed profiling mode not supported")
                except Exception as e:
                    log.debug(f"Could not enable profiling: {e}")
            else:
                try:
                    self.con.execute("PRAGMA disable_profiling")
                    if is_debug():
                        log.debug("DuckDB profiling disabled (no profiling output directory specified).")
                    else:
                        log.debug("DuckDB profiling disabled (not in debug mode).")
                except Exception as e:
                    log.debug(f"Could not disable profiling: {e}")

        except Exception as e:
            log.error(f"Error during basic DuckDB configuration: {e}")
            raise

    def close(self):
        """Close the database connection."""
        if self.con:
            self.con.close()
            self.con = None

    def execute(self, query, params=None):
        """Execute a query with optional parameters and capture profiling output."""
        if not self.con:
            self.connect()

        try:
            self.query_count += 1

            if self.profiling_enabled:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                operation_type = self._get_query_operation(query)
                target_table = self._get_target_table(query)

                if target_table:
                    friendly_name = f"{operation_type}_{target_table}"
                else:
                    query_short = query.strip().split("\n")[0][:30].replace(" ", "_").replace("/", "_")
                    friendly_name = f"{operation_type}_{query_short}"

                friendly_name = re.sub(r'[^\w\-]', '_', friendly_name)
                friendly_name = re.sub(r'_+', '_', friendly_name)
                friendly_name = friendly_name[:50]

                file_prefix = f"{timestamp}_q{self.query_count:03d}"
                query_safe_name = f"{file_prefix}_{friendly_name}"

                try:
                    sql_filepath = os.path.join(self.profile_output_dir, f"{query_safe_name}.sql")
                    with open(sql_filepath, 'w') as f:
                        f.write(query)
                    log.debug(f"Saved query to {sql_filepath}")
                except Exception as e:
                    log.debug(f"Failed to save query to file: {e}")

            if params:
                return self.con.execute(query, params)
            else:
                return self.con.execute(query)

        except duckdb.OutOfMemoryException as e:
            log.error(f"Out of memory error during query execution: {e}")
            log.error(f"Current memory limit: {self.memory_limit}")
            log.error(f"Consider reducing batch size or increasing memory limit")
            log.debug(f"Query that failed: {query[:200]}..." if len(query) > 200 else query)
            raise
        except Exception as e:
            log.error(f"Error executing query: {e}")
            log.debug(f"Query: {query}")
            if params:
                log.debug(f"Parameters: {params}")
            raise

    def _get_query_operation(self, query):
        """Extract the primary operation from a SQL query."""
        query_upper = query.upper().strip()

        if query_upper.startswith("SELECT"):
            return "SELECT"
        elif query_upper.startswith("CREATE TABLE") or query_upper.startswith("CREATE TEMP TABLE"):
            return "CREATE_TABLE"
        elif query_upper.startswith("CREATE VIEW") or query_upper.startswith("CREATE TEMPORARY VIEW"):
            return "CREATE_VIEW"
        elif query_upper.startswith("INSERT"):
            return "INSERT"
        elif query_upper.startswith("UPDATE"):
            return "UPDATE"
        elif query_upper.startswith("DELETE"):
            return "DELETE"
        elif query_upper.startswith("COPY"):
            return "COPY"
        elif query_upper.startswith("DROP"):
            return "DROP"
        elif query_upper.startswith("ANALYZE"):
            return "ANALYZE"
        elif query_upper.startswith("EXPLAIN"):
            return "EXPLAIN"
        else:
            first_word = query_upper.split()[0] if query_upper.split() else "QUERY"
            return first_word

    def _get_target_table(self, query):
        """Extract the target table name from a SQL query."""
        query_upper = query.upper().strip()

        table_patterns = [
            r"CREATE\s+(?:TEMP|TEMPORARY)?\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[\"']?(\w+)[\"']?",
            r"FROM\s+[\"']?(\w+)[\"']?",
            r"INSERT\s+INTO\s+[\"']?(\w+)[\"']?",
            r"UPDATE\s+[\"']?(\w+)[\"']?",
            r"DELETE\s+FROM\s+[\"']?(\w+)[\"']?",
            r"COPY\s+[\"']?(\w+)[\"']?",
            r"DROP\s+(?:TABLE|VIEW)\s+(?:IF\s+EXISTS\s+)?[\"']?(\w+)[\"']?",
            r"ANALYZE\s+[\"']?(\w+)[\"']?",
            r"EXPLAIN\s+.*\s+FROM\s+[\"']?(\w+)[\"']?"
        ]

        for pattern in table_patterns:
            match = re.search(pattern, query_upper)
            if match:
                return match.group(1).lower()

        temp_view_match = re.search(r"CREATE\s+(?:TEMP|TEMPORARY)?\s+VIEW\s+[\"']?(\w+)[\"']?", query_upper)
        if temp_view_match:
            return temp_view_match.group(1).lower()

        return None

    def _save_profiling_output(self, name):
        """Save the current profiling data to a file."""
        if not self.profiling_enabled or not os.path.isdir(self.profile_output_dir):
            return

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        profile_filename = f"{timestamp}_{name}.profile.json"
        profile_path = os.path.join(self.profile_output_dir, profile_filename)

        try:
            stats = {}

            try:
                table_info = self.con.execute("PRAGMA show_tables").fetchall()
                tables = [row[0] for row in table_info]

                table_stats = []
                for table in tables:
                    try:
                        row_count = self.con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                        table_stats.append({
                            "name": table,
                            "row_count": row_count
                        })
                    except:
                        pass

                stats["tables"] = table_stats
            except:
                pass

            try:
                memory_info = self.con.execute("PRAGMA memory_usage").fetchall()
                memory_stats = {}
                for row in memory_info:
                    if len(row) >= 2:
                        memory_stats[row[0]] = row[1]
                stats["memory_usage"] = memory_stats
            except:
                pass

            try:
                db_info = self.con.execute("PRAGMA database_size").fetchone()
                if db_info:
                    stats["database_size"] = db_info[0]
            except:
                pass

            # Add system resource information
            try:
                stats["system_info"] = {
                    "available_memory_gb": psutil.virtual_memory().available // (1024**3),
                    "cpu_count": os.cpu_count(),
                    "configured_threads": self.threads,
                    "configured_memory_limit": self.duckdb_memory_limit
                }
            except:
                pass

            profile_data = {
                "marker_name": name,
                "timestamp": timestamp,
                "query_count": self.query_count,
                "statistics": stats,
                "status": "profiling_active" if is_debug() else "profiling_inactive",
                "database_file": self.database
            }

            with open(profile_path, 'w') as f:
                json.dump(profile_data, f, indent=2)

            log.debug(f"Saved database statistics to {profile_path}")
            
            # Also display key profiling info on screen
            if name == "final_profiling":
                print(f"📊 Final profiling saved: {self.query_count} queries executed")
                if "tables" in stats:
                    total_rows = sum(t.get("row_count", 0) for t in stats["tables"])
                    print(f"📊 Total rows processed: {total_rows:,}")

        except Exception as e:
            log.debug(f"Error creating profiling statistics: {e}")

    def set_memory_limit(self, memory_limit):
        """Update memory limit."""
        if self.con:
            try:
                # Validate the new limit first
                validated_limit = self._validate_memory_limit(memory_limit)
                self.con.execute(f"SET memory_limit='{validated_limit}'")
                self.duckdb_memory_limit = validated_limit  # Update the stored limit
                log.info(f"Updated memory limit to {validated_limit}")
                return True
            except Exception as e:
                log.error(f"Failed to update memory limit: {e}")
                return False
        return False

    def table_exists(self, table_name):
        """Check if a table exists in the database."""
        try:
            result = self.con.execute(
                f"SELECT count(*) FROM information_schema.tables WHERE table_name='{table_name}'"
            ).fetchone()
            return result[0] > 0
        except Exception:
            try:
                self.con.execute(f"SELECT * FROM {table_name} LIMIT 0")
                return True
            except Exception:
                return False

    def get_table_row_count(self, table_name):
        """Get row count for a table."""
        try:
            result = self.con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
            return result[0] if result else 0
        except Exception as e:
            log.debug(f"Error getting row count for table '{table_name}': {e}")
            return 0

    def _validate_memory_limit(self, memory_limit):
        """Validate and format memory limit string."""
        if isinstance(memory_limit, (int, float)):
            # Handle numeric input - assume it's in bytes if very large, GB otherwise
            if memory_limit > 1024 * 1024 * 1024:  # If > 1GB in bytes
                # Convert bytes to GB
                memory_gb = max(1, int(memory_limit / (1024 ** 3)))
                log.debug(f"Converted {memory_limit} bytes to {memory_gb}GB")
                memory_limit = f"{memory_gb}GB"
            else:
                # Assume it's already in GB
                memory_limit = f"{int(memory_limit)}GB"

        if isinstance(memory_limit, str):
            # Remove any whitespace
            memory_limit = memory_limit.strip()

            # If no unit is specified, assume GB but validate it's reasonable
            if memory_limit.isdigit():
                num_val = int(memory_limit)
                available_mem_gb = psutil.virtual_memory().available // (1024 * 1024 * 1024)

                if num_val > available_mem_gb:
                    # If specified limit exceeds available memory, cap it
                    reasonable_limit = max(1, int(available_mem_gb * 0.8))
                    log.warning(f"Specified memory limit {num_val}GB exceeds available memory. Setting to {reasonable_limit}GB")
                    return f"{reasonable_limit}GB"

                return f"{memory_limit}GB"

            # Check if it ends with valid unit (support both single and multi-letter units)
            valid_units = {
                'B': 1, 'K': 1024, 'M': 1024**2, 'G': 1024**3, 'T': 1024**4,
                'KB': 1024, 'MB': 1024**2, 'GB': 1024**3, 'TB': 1024**4,
                'KIB': 1024, 'MIB': 1024**2, 'GIB': 1024**3, 'TIB': 1024**4
            }

            memory_upper = memory_limit.upper()

            # Try to match units - check longer units first to avoid partial matches
            for unit_name in sorted(valid_units.keys(), key=len, reverse=True):
                if memory_upper.endswith(unit_name):
                    # Extract the numeric part and validate
                    numeric_part = memory_upper[:-len(unit_name)].strip()
                    try:
                        num_val = float(numeric_part)

                        # Convert to GB for validation
                        bytes_val = num_val * valid_units[unit_name]
                        gb_val = bytes_val / (1024**3)

                        available_mem_gb = psutil.virtual_memory().available // (1024 * 1024 * 1024)
                        if gb_val > available_mem_gb:
                            reasonable_limit = max(1, int(available_mem_gb * 0.8))
                            log.warning(f"Specified memory limit {memory_limit} exceeds available memory. Setting to {reasonable_limit}GB")
                            return f"{reasonable_limit}GB"

                        # Normalize to GB format for DuckDB
                        if unit_name in ['G', 'GB', 'GIB']:
                            return f"{int(num_val)}GB"
                        elif unit_name in ['M', 'MB', 'MIB']:
                            gb_equivalent = max(1, int(num_val / 1024))
                            return f"{gb_equivalent}GB"
                        elif unit_name in ['K', 'KB', 'KIB']:
                            gb_equivalent = max(1, int(num_val / (1024 * 1024)))
                            return f"{gb_equivalent}GB"
                        elif unit_name in ['T', 'TB', 'TIB']:
                            gb_equivalent = int(num_val * 1024)
                            return f"{gb_equivalent}GB"
                        else:  # bytes
                            gb_equivalent = max(1, int(num_val / (1024**3)))
                            return f"{gb_equivalent}GB"

                    except ValueError:
                        break

            # If it ends with just a number, add GB
            if memory_limit[-1].isdigit():
                return f"{memory_limit}GB"

        # Default fallback - use 80% of available memory
        available_mem_gb = psutil.virtual_memory().available // (1024 * 1024 * 1024)
        fallback_limit = max(1, int(available_mem_gb * 0.8))
        log.warning(f"Invalid memory limit format: {memory_limit}. Using {fallback_limit}GB (80% of available memory).")
        return f"{fallback_limit}GB"
