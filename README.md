# PGAI

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
