# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 11:29:26 2025

@author: Lenovo
"""

import pandas as pd 
import numpy as np 
from surprise import Reader, Dataset, KNNBasic, accuracy
from surprise.model_selection import train_test_split
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List
import schemas
import models
from models import get_db
from sqlalchemy.orm import Session
from sqlalchemy.sql import func
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict


app = FastAPI()


 



n_users = 100
n_products = 300
n_interactions = 1000

user_ids = [f"user_{i}" for i in range(1, n_users + 1)]
product_ids = [f"product_{i}" for i in range(1, n_products + 1)]

interactions = []

for _ in range(n_interactions):
    user_id = np.random.choice(user_ids)
    product_id = np.random.choice(product_ids)
    rating = np.random.randint(1, 6)
    interactions.append({'user_id':user_id, 'product_id':product_id, 'rating':rating})
    
df = pd.DataFrame(interactions)

print(f"Number of users:{len(df['user_id'].unique())}")
print(f'Number of products:{len(df['product_id'].unique())}')
print(f"Number of interactions:{len(df)}")

print("\nSample of interactions:")
print(df.head())
df.to_csv('interactions.csv')


df = pd.read_csv('interactions.csv')
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['user_id','product_id', 'rating']], reader)

trainset , testset = train_test_split(data, test_size=0.25)

sim_option = {
    'name':'cosine',
    'user_based':True}

algo = KNNBasic(sim_options=sim_option)
algo.fit(trainset)

predictions = algo.test(testset)
accuracy.rmse(predictions)
accuracy.mae(predictions)

def get_top_n_recommendation(user_id, n):
    all_product_ids = df['product_id'].unique()
    user_product_ids = df[df['user_id']==user_id]['product_id'].tolist()
    product_ids_to_predict = [pid for pid in all_product_ids if pid not in user_product_ids]
    predictions = [algo.predict(user_id, pid) for pid in product_ids_to_predict]
    predictions.sort(key=lambda x:x.est, reverse=True)
    top_n_prediction = [pred.iid for pred in predictions[:n]]
    return top_n_prediction


    
    

    
    
    
items = {
    'item_1':
        {
        'item_id':'item_1', 
         'name':'Hammer', 
         'description':'A sturdy Hammer'},
    'item_2':
        {
        'item_id':'item_2', 
        'name':'ScrewDriver', 
        'description':'A reliable ScrewDriver'},
    'item_3':
        {
            'item_id':'item_3', 
            'name':'Wrench', 
            'description':'An adjustable wrench'}}
    

@app.on_event('startup')
async def load_intial_data():
    db = get_db().__next__()
    try:
        if db.query(models.User).count() == 0:
            df = pd.read_csv('interactions.csv')
            for user_id in df['user_id'].unique():
                db_user = models.User(user_id = user_id , name="Default Name", email = "default@gmail.com")
                db.add(db_user)
            for product_id in df['product_id'].unique():
                db_item = models.Item(item_id = product_id, name = "Default Item", description = "default description")
                db.add(db_item)
            db.commit()
    finally:
        db.close()
    

    
@app.get('/recommendations/{user_id}', response_model=List[schemas.Item])
async def get_recommendations(user_id: str, db: Session = Depends(models.get_db)):
    # 1️⃣ Check if user exists in DB
    user = db.query(models.User).filter(models.User.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")

    # 2️⃣ Load interactions from DB
    interactions = db.query(models.UserItemInteraction).all()
    interaction_list = [
        {'user_id': interaction.user_id, 'item_id': interaction.item_id, 'rating': interaction.rating}
        for interaction in interactions
    ]
    df = pd.DataFrame(interaction_list)

    # 3️⃣ Handle case where user has no interactions yet
    if df.empty or user_id not in df['user_id'].values:
        # Return top 10 popular items as default
        top_items = df.groupby('item_id')['rating'].sum().sort_values(ascending=False).head(10).index.tolist()
        return db.query(models.Item).filter(models.Item.item_id.in_(top_items)).all()

    # 4️⃣ Build user-item matrix
    user_item_matrix = df.pivot_table(index='user_id', columns='item_id', values='rating', fill_value=0)

    # 5️⃣ Compute user-user similarity
    user_similarity = cosine_similarity(user_item_matrix)
    similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

    # 6️⃣ Find top similar users
    similar_users = similarity_df[user_id].sort_values(ascending=False).drop(user_id).head(10)

    # 7️⃣ Aggregate items from similar users
    recommended_scores = defaultdict(float)
    for sim_user_id, similarity_score in similar_users.items():
        user_items = df[df['user_id'] == sim_user_id][['item_id', 'rating']]
        for _, row in user_items.iterrows():
            recommended_scores[row['item_id']] += row['rating'] * similarity_score

    # 8️⃣ Remove items the user already interacted with
    user_items_set = set(df[df['user_id'] == user_id]['item_id'])
    for item in user_items_set:
        recommended_scores.pop(item, None)

    # 9️⃣ Select top 10 recommended items
    top_items = sorted(recommended_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    top_item_ids = [item_id for item_id, _ in top_items]

    # 🔟 Fetch item details from DB
    recommendations = db.query(models.Item).filter(models.Item.item_id.in_(top_item_ids)).all()
    return recommendations
    
        

@app.get('/items/{item_id}', response_model=schemas.Item)
async def get_item(item_id:str, db: Session = Depends(get_db)):
    item = db.query(models.Item).filter(models.Item.item_id == item_id).first()
    if item:
        return item
    else:
        raise HTTPException(status_code=404 , detail='Items not found')
        
@app.get('/items/', response_model=List[schemas.Item])
async def get_list_items(db:Session = Depends(get_db)):
    items = db.query(models.Item).all()
    return items

@app.post('/items/', response_model=schemas.Item)
async def create_item(item:schemas.CreateItem, db: Session= Depends(get_db)):
    db_item = models.Item(**item.dict())
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item

@app.put('/items/{item_id}', response_model=schemas.Item)
async def update_item(item_id : str, item: schemas.CreateItem, db: Session = Depends(get_db)):
    db_item = db.query(models.Item).filter(models.Item.item_id == item_id).first()
    if db_item:
        for key ,  value in item.dict().items():
            setattr(db_item, key, value)
        db.commit()
        db.refresh(db_item)
        return db_item
    else:
        raise HTTPException(status_code=404, detail='item not found')

@app.delete('/items/{item_id}')
async def delete_item(item_id:str, db:Session = Depends(get_db)):
    db_item = db.query(models.Item).filter(models.Item.item_id == item_id).first()
    if db_item:
        db.delete(db_item)
        db.commit()
        db.refresh(db_item)
        return {'message':'item deleted'}
    else:
        raise HTTPException(status_code=404, detail='item not found')
        

@app.post('/users/', response_model=schemas.User)
async def create_user(user:schemas.CreateUser, db: Session=Depends(get_db)):
    db_user = models.User(**user.dict())
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user
@app.get('/users/{user_id}', response_model=schemas.User)
async def get_user(user_id:str, db:Session=Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.user_id == user_id).first()
    if db_user:
        return db_user
    else:
        raise HTTPException(status_code=404, detail='user not found')

@app.get('/users/', response_model=List[schemas.User])
async def get_list_users(db:Session=Depends(get_db)):
    db_user = db.query(models.User).all()
    if db_user:
        return db_user
    else:
        raise HTTPException(status_code=404, detail='user not found')
        
@app.put('/users/{user_id}', response_model=schemas.User)
async def update_user(user_id:str, user:schemas.CreateUser, db:Session=Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.user_id == user_id).first()
    if db_user:
        for key ,  value in user.dict().items():
            setattr(db_user, key, value)
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        return db_user
    else:
        raise HTTPException(status_code=404, detail='user not found')

@app.delete('/users/{user_id}')
async def delete_user(user_id:str, db:Session=Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.user_id == user_id).first()
    if db_user:
        db.delete(db_user)
        db.commit()
        db.refresh()
        return {'message':'user deleted'}
    else:
        raise HTTPException(status_code=404, detail='user not found')
        
@app.post('/users/{user_id}/interactions', response_model=schemas.UserItemInteraction)
async def create_interaction(user_id:int, interaction:schemas.InteractionCreate , db:Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.id == user_id).first()
    db_item = db.query(models.Item).filter(models.Item.id == interaction.item_id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail='user not found')
    if not db_item:
        raise HTTPException(status_code=404, detail='item not found')
    db_interaction = models.UserItemInteraction(user_id = user_id, item_id = interaction.item_id, interaction_type = interaction.interaction_type, rating = interaction.rating)
    db.add(db_interaction)
    db.commit()
    db.refresh(db_interaction)
    return db_interaction


user_id = 'user_1'
n = 10
recommendation = get_top_n_recommendation(user_id, n)
print('\nTop {n} recommendation of {user_id}')
print(recommendation)