import sqlite3

def init_db():
    conn = sqlite3.connect("quickship.db")
    cursor = conn.cursor()
    
    # Create Tables
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Users (
        user_id TEXT PRIMARY KEY,
        name TEXT,
        phone TEXT
    )''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Orders (
        order_id TEXT PRIMARY KEY,
        user_id TEXT,
        status TEXT,
        amount REAL,
        address TEXT,
        FOREIGN KEY(user_id) REFERENCES Users(user_id)
    )''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Drivers (
        driver_id TEXT PRIMARY KEY,
        status TEXT,
        location TEXT
    )''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Payments (
        order_id TEXT PRIMARY KEY,
        refund_status TEXT,
        FOREIGN KEY(order_id) REFERENCES Orders(order_id)
    )''')
    
    # Insert Mock Data for Testing
    cursor.executemany("INSERT OR IGNORE INTO Users VALUES (?, ?, ?)", [
        ("U101", "Alice Smith", "555-0192"),
        ("U102", "Bob Jones", "555-0143")
    ])
    cursor.executemany("INSERT OR IGNORE INTO Orders VALUES (?, ?, ?, ?, ?)", [
        ("ORD9901", "U101", "In Transit", 45.99, "123 Maple St"),
        ("ORD9902", "U102", "Delivered", 120.00, "782 Oak Ave")
    ])
    cursor.executemany("INSERT OR IGNORE INTO Drivers VALUES (?, ?, ?)", [
        ("DRV01", "Available", "Hub A"),
        ("DRV02", "Busy", "Route 4")
    ])
    cursor.executemany("INSERT OR IGNORE INTO Payments VALUES (?, ?)", [
        ("ORD9901", "Not Requested"),
        ("ORD9902", "Pending")
    ])
    
    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()
    print("Database initialized successfully.")