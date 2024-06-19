# PostgreSQL Installation Guide

This guide provides step-by-step instructions to install PostgreSQL on various operating systems.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation on Ubuntu](#installation-on-ubuntu)
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



# pgai Extension for PostgreSQL

pgai is a PostgreSQL extension that fetches predictive values from a trained model and places them in pseudo columns for future dates.

## Installation Instructions

1. **Prerequisites:**  
   Ensure you have PostgreSQL installed. Download from the official [PostgreSQL website](https://www.postgresql.org/download/).

2. **Cloning the Repository:**  
   Clone the pgai repository and navigate to the directory:
   ```bash
   git clone https://github.com/yourusername/pgai.git
   cd pgai
   ```

3. **Export PostgreSQL Path:**  
   Export the PostgreSQL bin directory to your PATH. Replace `/path/to/postgres` with your PostgreSQL installation path:
   ```bash
   export PATH=/path/to/postgres/bin:$PATH
   ```

4. **Run PostgreSQL:**  
   Ensure PostgreSQL is running:
   ```bash
   pg_ctl start
   ```

5. **Create the Extension:**  
   Connect to your PostgreSQL instance and create the pgai extension:
   ```sql
   CREATE EXTENSION pgai;
   ```

## Usage

Start using the pgai functions and features in your PostgreSQL queries.

## Contributing

We welcome contributions. Fork the repository and submit pull requests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or feedback, please open an issue on the GitHub repository or contact us at your.email@example.com.

---

Happy Predicting with pgai!

