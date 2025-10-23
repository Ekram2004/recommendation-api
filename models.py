# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 07:11:14 2025

@author: Lenovo
"""

from sqlalchemy import create_engine, Column, Integer, String , Float, ForeignKey, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func


DATABASE_URL='postgresql://postgres:ekrampostgres@localhost/recommendations'
engine = create_engine(DATABASE_URL)
Base = declarative_base()

class Item(Base):
    __tablename__ = 'items'
    id = Column(Integer, primary_key=True, index=True)
    item_id = Column(String, unique=True, index=True)
    name = Column(String)
    description = Column(String)
    

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, unique=True, index=True)
    name = Column(String)
    email = Column(String)
    
    
class UserItemInteraction(Base):
    __tablename__ = 'user_item_interactions'
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    item_id = Column(Integer, ForeignKey('items.id'))
    interaction_type = Column(String(255))
    rating = Column(Float, nullable=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    
    user = relationship('User', back_populates='item_interactions')
    item = relationship('Item', back_populates='user_interactions')
    
User.item_interactions = relationship('UserItemInteraction', back_populates='user')
Item.user_interactions = relationship('UserItemInteraction', back_populates='item')
    
    
Base.metadata.create_all(bind=engine)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        

