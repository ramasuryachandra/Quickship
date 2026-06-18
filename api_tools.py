import sqlite3

DB_PATH = "quickship.db"


def query_db(query, params=(), one=False, commit=False):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(query, params)
    if commit:
        conn.commit()
        conn.close()
        return True
    res = cursor.fetchall()
    conn.close()
    return (res[0] if res else None) if one else res


def get_order_status(order_id: str) -> dict:
    res = query_db("SELECT status, address FROM Orders WHERE order_id = ?", (order_id,), one=True)
    if not res:
        return {"error": f"Order '{order_id}' not found."}
    return {"order_id": order_id, "status": res[0], "address": res[1]}


def update_delivery_status(order_id: str, status: str) -> dict:
    valid_statuses = ["Pending", "In Transit", "Delivered", "Cancelled"]
    if status not in valid_statuses:
        return {"error": f"Invalid status '{status}'. Allowed: {', '.join(valid_statuses)}"}
    query_db("UPDATE Orders SET status = ? WHERE order_id = ?", (status, order_id), commit=True)
    return {"message": f"Order {order_id} updated to {status}."}


def issue_refund(order_id: str) -> dict:
    order = query_db("SELECT status FROM Orders WHERE order_id = ?", (order_id,), one=True)
    if not order:
        return {"error": f"Order '{order_id}' not found."}
    query_db("INSERT OR REPLACE INTO Payments (order_id, refund_status) VALUES (?, 'Refunded')", (order_id,), commit=True)
    return {"order_id": order_id, "refund_status": "Refunded"}


def assign_driver(order_id: str) -> dict:
    driver = query_db("SELECT driver_id FROM Drivers WHERE status = 'Available' LIMIT 1", one=True)
    if not driver:
        return {"error": "No available drivers at the moment."}
    query_db("UPDATE Drivers SET status = 'Busy' WHERE driver_id = ?", (driver[0],), commit=True)
    return {"order_id": order_id, "assigned_driver_id": driver[0]}


def get_user_details(user_id: str) -> dict:
    res = query_db("SELECT name, phone FROM Users WHERE user_id = ?", (user_id,), one=True)
    if not res:
        return {"error": f"User '{user_id}' not found."}
    return {"user_id": user_id, "name": res[0], "phone": res[1]}