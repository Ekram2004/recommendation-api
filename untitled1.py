# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 10:26:49 2025

@author: Lenovo
"""


from fastapi import Depends, HTTPException, FastAPI
from sqlalchemy.orm import Session
from typing import List
from . import models, schemas
from .config import get_db
from pydantic import BaseModel
import models as _models, schemas as _schemas
import pandas as pd  # Import pandas
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import func

app = FastAPI()

_models.Base.metadata.create_all(bind=engine)

# 1. Load data from CSV to insert at database if the database is empty
@app.on_event("startup")
async def load_initial_data():
    db = get_db().__next__()  # Get a database session
    try:
        # Check if there are already users
        if db.query(models.User).count() == 0:
            df = pd.read_csv("interactions.csv")  # Load your CSV
            # Iterate over unique users
            for user_id in df["user_id"].unique():
                db_user = models.User(user_id=user_id, name="Default Name", email="default@example.com")
                db.add(db_user)
            # Iterate over unique items
            for product_id in df["product_id"].unique():
                db_item = models.Item(item_id=product_id, name="Default Item", description="Default Description")
                db.add(db_item)
            db.commit()  # Commit after adding all users and items
    finally:
        db.close()  # Ensure the database session is closed

# 2. Create Item API
@app.post("/items/", response_model=schemas.Item)
async def create_item(item: schemas.CreateItem, db: Session = Depends(get_db)):
    db_item = models.Item(**item.dict())
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item

# 3. Get Recommendation by User ID
@app.get("/recommendations/{user_id}", response_model=List[schemas.Item])
async def get_recommendations(user_id: str, db: Session = Depends(get_db)):
    # 1. Get the user by user_id
    user = db.query(models.User).filter(models.User.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # 2. Get all items
    items = db.query(models.Item).all()

    # 3. Create a dictionary to store the predicted ratings for each item
    predicted_ratings = {}

    # 4. Loop through the items and predict the rating for the current user
    for item in items:
        # Calculate a score based on other users' interactions and Item
        # - Get others interactions for this user for the current item
        query = db.query(
            models.UserItemInteraction.user_id,
            func.count(models.UserItemInteraction.item_id).label('interaction_count')
        ).filter(
            models.UserItemInteraction.item_id == item.id,
            models.UserItemInteraction.user_id != user.id  # Exclude current user
        ).group_by(models.UserItemInteraction.user_id).order_by(func.count(models.UserItemInteraction.item_id).desc())

        similar_users = query.limit(5).all()
        score = 0
        for similar_user in similar_users:
            score = score + 1 # Add 1 score for each similar user

        predicted_ratings[item.id] = score

    # 5. Sort the items by predicted rating in descending order
    sorted_items = sorted(predicted_ratings.items(), key=lambda x: x[1], reverse=True)

    # 6. Get the top 10 recommended item IDs
    top_item_ids = [item_id for item_id, rating in sorted_items[:10]]

    # 7. Retrieve the corresponding `Item` objects from the database
    recommendations = db.query(models.Item).filter(models.Item.id.in_(top_item_ids)).all()

    # 8. Return the recommended items
    return recommendations

# 4. Get Item by Item ID
@app.get("/items/{item_id}", response_model=schemas.Item)
async def get_item(item_id: str, db: Session = Depends(get_db)):
    item = db.query(models.Item).filter(models.Item.item_id == item_id).first()
    if item:
        return item
    else:
        raise HTTPException(status_code=404, detail="Item not found")

# 5. Get all Items API
@app.get("/items/", response_model=List[schemas.Item])
async def get_list_item(db: Session = Depends(get_db)):
    items = db.query(models.Item).all()
    return items


# 6. Update Item API
@app.put("/items/{item_id}", response_model=schemas.Item)
async def update_item(item_id: str, item: schemas.CreateItem, db: Session = Depends(get_db)):
    db_item = db.query(models.Item).filter(models.Item.item_id == item_id).first()
    if db_item:
        for key, value in item.dict().items():
            setattr(db_item, key, value)
        db.commit()
        db.refresh(db_item)
        return db_item
    else:
        raise HTTPException(status_code=404, detail="Item not found")

# 7. Delete Item API
@app.delete("/items/{item_id}")
async def delete_item(item_id: str, db: Session = Depends(get_db)):
    item = db.query(models.Item).filter(models.Item.item_id == item_id).first()
    if item:
        db.delete(item)
        db.commit()
        return {"message": "Item deleted"}
    else:
        raise HTTPException(status_code=404, detail="Item not found")


# 8. Create User API
@app.post("/users/", response_model=schemas.User)
async def create_user(user: schemas.CreateUser, db: Session = Depends(get_db)):
    db_user = models.User(**user.dict())
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

# 9. Get User by User ID
@app.get("/users/{user_id}", response_model=schemas.User)
async def get_user(user_id: str, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.user_id == user_id).first()
    if user:
        return user
    else:
        raise HTTPException(status_code=404, detail="User not found")

# 10. Update User API
@app.put("/users/{user_id}", response_model=schemas.User)
async def update_user(user_id: str, user: schemas.CreateUser, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.user_id == user_id).first()
    if db_user:
        for key, value in user.dict().items():
            setattr(db_user, key, value)
        db.commit()
        db.refresh(db_user)
        return db_user
    else:
        raise HTTPException(status_code=404, detail="User not found")

# 11. Delete User API
@app.delete("/users/{user_id}")
async def delete_user(user_id: str, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.user_id == user_id).first()
    if user:
        db.delete(user)
        db.commit()
        return {"message": "User deleted"}
    else:
        raise HTTPException(status_code=404, detail="User not found")


# 12. Create Interaction Tracking API
@app.post("/users/{user_id}/interactions", response_model=schemas.UserItemInteraction)
async def create_interaction(
    user_id: int,
    interaction: schemas.InteractionCreate,
    db: Session = Depends(get_db)
):
    """
    Creates a new user-item interaction record.
    """
    # Check if the user and item exist
    db_user = db.query(models.User).filter(models.User.id == user_id).first()
    db_item = db.query(models.Item).filter(models.Item.id == interaction.item_id).first()

    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    if not db_item:
        raise HTTPException(status_code=404, detail="Item not found")

    # Create the interaction
    db_interaction = models.UserItemInteraction(
        user_id=user_id,
        item_id=interaction.item_id,
        interaction_type=interaction.interaction_type,
        rating=interaction.rating
    )

    db.add(db_interaction)
    db.commit()
    db.refresh(db_interaction)

    return db_interaction
