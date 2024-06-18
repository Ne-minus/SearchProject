import os
import pandas as pd
from sqlalchemy import Column, Integer, Text, String, create_engine, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, aliased, Query

engine = create_engine(f"postgresql://ne-minus:220100es@db:5432/polit_news", echo=True)
Base = declarative_base()
Session = sessionmaker(bind=engine)

class Texts(Base):
    __tablename__ = "texts"
    index = Column('index', Integer, primary_key=True)
    text_id = Column('text_id', Text)
    url = Column('url', Text)
    title = Column('title', Text)
    text = Column('text', Text)
    topic = Column('topic', Text)
    tags = Column('tags', Text)
    date = Column('date', Text)
    clean = Column('clean', Text)
    clean_pos = Column('clean_pos', Text)

Base.metadata.create_all(engine)

# if not sqlalchemy.inspect(engine).has_table("texts"):
# texts = pd.read_csv('isdb_hw2.csv')
# texts = texts.reset_index()
#
# texts.to_sql('texts', con=engine, index=True, if_exists='replace')

