# database.py
import sqlite3
import pandas as pd
import os
from datetime import datetime

class DatabaseManager:
    def __init__(self, db_path="./data/clinixbot.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """初始化数据库"""
        # 检查数据库文件夹是否存在
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # 连接数据库
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建用户表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            name TEXT,
            email TEXT,
            dob TEXT,
            gender TEXT,
            created_at TEXT
        )
        ''')
        
        # 创建对话历史表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            chat_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            timestamp TEXT,
            user_message TEXT,
            bot_response TEXT,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
        ''')
        
        # 创建诊断记录表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS diagnoses (
            diagnosis_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            chat_id INTEGER,
            symptoms TEXT,
            diagnosis TEXT,
            recommended_treatments TEXT,
            timestamp TEXT,
            FOREIGN KEY (user_id) REFERENCES users (user_id),
            FOREIGN KEY (chat_id) REFERENCES chat_history (chat_id)
        )
        ''')
        
        # 保存更改并关闭连接
        conn.commit()
        conn.close()
    
    def add_user(self, user_id, name, email, dob, gender):
        """添加新用户"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
            INSERT INTO users (user_id, name, email, dob, gender, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (user_id, name, email, dob, gender, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            
            conn.commit()
            return True
        except Exception as e:
            print(f"添加用户时出错: {e}")
            return False
        finally:
            conn.close()
    
    def get_chat_history(self, user_id, limit=10):
        """获取用户聊天历史"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
            SELECT * FROM chat_history 
            WHERE user_id = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
            ''', (user_id, limit))
            
            history = cursor.fetchall()
            return [dict(row) for row in history]
        except Exception as e:
            print(f"获取聊天历史时出错: {e}")
            return []
        finally:
            conn.close()