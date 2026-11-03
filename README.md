# BVMT AI Trader

An AI-driven trading system specialized in the Tunisian Stock Exchange (Bourse de Tunis).

## Windows Installation & Setup Guide

### 1. Prerequisites
- **Python 3.12+** installed and added to your system PATH.
- **PostgreSQL** installed and running locally.

### 2. Clone and Setup Environment
Open **Command Prompt (cmd)** or **PowerShell** in your workspace directory:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows Command Prompt)
venv\Scripts\activate.bat

# Or for PowerShell:
# venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### 3. Database Configuration
1. Open pgAdmin or your SQL client and create a new PostgreSQL database:
   ```sql
   CREATE DATABASE bvmt_ai_trader;
   ```
2. Copy `.env.example` to `.env`:
   ```bash
   copy .env.example .env
   ```
3. Update `.env` with your PostgreSQL username and password:
   ```env
   DB_USER=postgres
   DB_PASSWORD=your_actual_password
   DB_HOST=localhost
   DB_PORT=5432
   DB_NAME=bvmt_ai_trader
   ```

### 4. Test the Database Connection & Schema
Run the connection test script to verify setup and create the tables:
```bash
python scripts/test_connection.py
```

## Competition Simulation

The BVMT competition benchmark starts with 10,000 TND by default and executes signals at the next available session open. It applies no broker commission or slippage: buys deduct only the share purchase amount, sells add only sale proceeds, and gross and net P&L are equal. Set a different initial capital through the benchmark runner API when needed.

Run the BT and SFBT benchmark pipeline with:

```bash
python -m src.models.benchmark_runner
```
