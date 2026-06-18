from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sqlite3

app = FastAPI(title="QuickShip Internal API Tools")

def query_db(query, params=(), one=False, commit=False):
    conn = sqlite3.connect("quickship.db")
    cursor = conn.cursor()
    cursor.execute(query, params)
    if commit:
        conn.commit()
        conn.close()
        return True
    res = cursor.fetchall()
    conn.close()
    return (res[0] if res else None) if one else res

@app.get("/order/{order_id}")
def get_order_status(order_id: str):
    res = query_db("SELECT status, address FROM Orders WHERE order_id = ?", (order_id,), one=True)
    if not res:
        raise HTTPException(status_code=404, detail="Order not found")
    return {"order_id": order_id, "status": res[0], "address": res[1]}

@app.post("/order/update")
def update_delivery_status(order_id: str, status: str):
    # Security input validation
    valid_statuses = ["Pending", "In Transit", "Delivered", "Cancelled"]
    if status not in valid_statuses:
        raise HTTPException(status_code=400, detail="Invalid status")
    
    query_db("UPDATE Orders SET status = ? WHERE order_id = ?", (status, order_id), commit=True)
    return {"message": f"Order {order_id} updated to {status}."}

@app.post("/order/refund")
def issue_refund(order_id: str):
    order = query_db("SELECT status FROM Orders WHERE order_id = ?", (order_id,), one=True)
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    
    query_db("INSERT OR REPLACE INTO Payments (order_id, refund_status) VALUES (?, 'Refunded')", (order_id,), commit=True)
    return {"order_id": order_id, "refund_status": "Refunded"}

@app.post("/driver/assign")
def assign_driver(order_id: str):
    driver = query_db("SELECT driver_id FROM Drivers WHERE status = 'Available' LIMIT 1", one=True)
    if not driver:
        raise HTTPException(status_code=404, detail="No available drivers")
    
    query_db("UPDATE Drivers SET status = 'Busy' WHERE driver_id = ?", (driver[0],), commit=True)
    return {"order_id": order_id, "assigned_driver_id": driver[0]}

@app.get("/user/{user_id}")
def get_user_details(user_id: str):
    res = query_db("SELECT name, phone FROM Users WHERE user_id = ?", (user_id,), one=True)
    if not res:
        raise HTTPException(status_code=404, detail="User not found")
    return {"user_id": user_id, "name": res[0], "phone": res[1]}