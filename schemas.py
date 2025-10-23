# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 07:17:55 2025

@author: Lenovo
"""

from pydantic import BaseModel
from sqlalchemy import Column, Integer, String , Float
from typing import List ,Optional

class CreateItem(BaseModel):
    item_id : str 
    name : str 
    description : str 
    
    class Config:
     from_attributes = True


class Item(BaseModel):
    id:int
    item_id : str 
    name: str 
    description : str
    
    

class CreateUser(BaseModel):
    user_id : str 
    name: str 
    email : str 
    
    class Config:
     from_attributes = True

    
class User(BaseModel):
    id: int 
    user_id : str
    name : str 
    email : str
    
    
class InteractionCreate(BaseModel):
    item_id : int 
    interaction_type : str 
    rating: Optional[float] = None
    
class UserItemInteraction(BaseModel):
    id: int 
    user_id : int 
    item_id : int
    interaction_type: str
    rating: Optional[float] = None