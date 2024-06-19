# PGAI

# PostgreSQL Installation Guide

This guide provides step-by-step instructions to install PostgreSQL on various operating systems.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation on Ubuntu](#installation-on-ubuntu)
- [Installation on Windows](#installation-on-windows)
- [Installation on macOS](#installation-on-macos)
- [Post-Installation Setup](#post-installation-setup)
- [Uninstallation](#uninstallation)
- [Troubleshooting](#troubleshooting)

## Prerequisites

- Ensure you have administrative privileges on your system.
- An internet connection to download necessary files.

## Installation on Ubuntu

1. Update the package list:
    ```sh
    sudo apt update
    ```

2. Install PostgreSQL:
    ```sh
    sudo apt install postgresql postgresql-contrib
    ```

3. Verify the installation:
    ```sh
    psql --version
    ```

## Installation on Windows

1. Download the installer from the [official PostgreSQL website](https://www.postgresql.org/download/windows/).

2. Run the installer and follow the on-screen instructions.

3. Choose the components you want to install. Ensure that the following components are selected:
    - PostgreSQL Server
    - pgAdmin 4
    - Command Line Tools

4. Set a password for the PostgreSQL superuser (postgres).

5. Complete the installation and verify by opening `pgAdmin` or running `psql` in the command prompt.

## Installation on macOS

1. Install Homebrew if you haven't already:
    ```sh
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    ```

2. Install PostgreSQL using Homebrew:
    ```sh
    brew install postgresql
    ```

3. Start the PostgreSQL service:
    ```sh
    brew services start postgresql
    ```

4. Verify the installation:
    ```sh
    psql --version
    ```

## Post-Installation Setup

### Creating a New User

1. Switch to the `postgres` user:
    ```sh
    sudo -i -u postgres
    ```

2. Open the PostgreSQL prompt:
    ```sh
    psql
    ```

3. Create a new user:
    ```sql
    CREATE USER yourusername WITH PASSWORD 'yourpassword';
    ```

4. Grant privileges to the new user:
    ```sql
    ALTER USER yourusername WITH SUPERUSER;
    ```

5. Exit the PostgreSQL prompt:
    ```sh
    \q
    ```

### Creating a New Database

1. Switch to the `postgres` user if not already done:
    ```sh
    sudo -i -u postgres
    ```

2. Create a new database:
    ```sh
    createdb yourdatabase
    ```

## Uninstallation

### Ubuntu

1. Remove PostgreSQL packages:
    ```sh
    sudo apt remove --purge postgresql postgresql-contrib
    ```

2. Remove data directories:
    ```sh
    sudo rm -rf /var/lib/postgresql/
    sudo rm -rf /etc/postgresql/
    sudo rm -rf /etc/postgresql-common/
    ```

### Windows

1. Open the Control Panel and navigate to `Programs and Features`.

2. Find PostgreSQL in the list of installed programs and click `Uninstall`.

### macOS

1. Stop the PostgreSQL service:
    ```sh
    brew services stop postgresql
    ```

2. Uninstall PostgreSQL:
    ```sh
    brew uninstall postgresql
    ```

3. Remove data directories:
    ```sh
    rm -rf /usr/local/var/postgres
    ```

## Troubleshooting

### Common Issues

- **Unable to connect to the server:** Ensure the PostgreSQL server is running and listening on the correct port.
- **Authentication failed:** Verify the username and password are correct.
- **Permission denied:** Ensure you have the necessary privileges to perform the desired action.

For more detailed troubleshooting, refer to the [PostgreSQL documentation](https://www.postgresql.org/docs/).

---

Feel free to contribute to this guide by submitting a pull request or opening an issue on our GitHub repository.

## Environment Variables for PostgreSQL

This project requires certain environment variables to be set in a `.env` file at the root of the project directory for PostgreSQL database configuration. Follow the steps below to create and configure your `.env` file.

### Steps to Create a `.env` File

1. **Create a new file named `.env`** in the root directory of your project. You can do this using your file explorer or from the terminal:
   ```bash
   touch .env

# PostgreSQL configuration
```bash
DB_HOST=localhost
DB_PORT=5432
DB_NAME=your_database_name
DB_USER=your_database_user
DB_PASSWORD=your_database_password


-- Create the db_config table to store database connection details
CREATE TABLE IF NOT EXISTS db_config (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Insert the database connection details into the db_config table
INSERT INTO db_config (key, value) VALUES
    ('db_host', 'localhost'),
    ('db_port', '5432'),
    ('db_name', 'postgres'),
    ('db_user', 'moizibrar'),
    ('db_password', 'postgres');


-- Create the model_config table to store model paths
CREATE TABLE IF NOT EXISTS model_config (
    model_name TEXT PRIMARY KEY,
    model_path TEXT NOT NULL
);

-- Insert the model path into the model_config table
INSERT INTO model_config (model_name, model_path) VALUES
    ('google_arima_model', '/Users/moizibrar/work/pgai/arima_model.h5');
