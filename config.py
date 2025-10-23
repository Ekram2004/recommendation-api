# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 07:18:02 2025

@author: Lenovo
"""

from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session
import models, schemas
from config import get_db, engine, Base
from typing import List

 # Create tables 
 
Base.metadata.create_all(bind=engine)

app = FastAPI(title='Recommendation API')

@app.post('/users/', response_model=schemas.User)
def create_user(user:schemas.CreateUser, db:Session = Depends(get_db)):
    db_user = models.User(**user.dict())
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user
    
@app.get('/users/', response_model=List[schemas.User])
def get_list_users(db:Session = Depends(get_db)):
    db_user = db.query(models.user).all()
    return db_user

@app.post('/items/', response_model=schemas.Item)
def create_item(item:schemas.CreateItem, db:Session = Depends(get_db)):
    db_item = models.Item(**item.dict())
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item

@app.get('/items/', response_model=List[schemas.Item])
def get_list_items(db: Session = Depends(get_db)):
    db_item = db.query(models.Item).all()
    return db_item