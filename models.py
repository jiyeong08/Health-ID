from sqlalchemy import Column, Integer, Boolean, String, DateTime
from sqlalchemy.sql import func
from database import Base
class Regist(Base):
    __tablename__ = 'regist'
    id = Column(String(6), primary_key=True)
    name = Column(String(50))
    regist_date = Column(DateTime, server_default=func.now())  # 등록일자 추가, 현재 시간으로 자동 설정
    file_path = Column(String(100))

class EnterExit(Base):
    __tablename__ = 'enterexit'
    no = Column(Integer, primary_key=True)
    id = Column(String(6))
    check_time = Column(DateTime, server_default=func.now()) # 현재 시간으로 자동 설정
    status = Column(String)