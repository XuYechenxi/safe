"""
数据库模块 - 使用SQLite存储用户数据和生成历史
"""
import sqlite3
import os
import threading
from datetime import datetime
from typing import Optional, List, Dict, Tuple
import hashlib
import json


class Database:
    """数据库管理类"""
    
    def __init__(self, db_path: str = "app.db"):
        """
        初始化数据库
        
        Args:
            db_path: 数据库文件路径
        """
        self.db_path = db_path
        self.init_database()
    
    def get_connection(self, timeout=5.0):
        """
        获取数据库连接（优化：降低超时时间，快速失败）
        
        Args:
            timeout: 连接超时时间（秒），默认5秒（降低以快速失败）
        
        Returns:
            数据库连接对象
        """
        try:
            conn = sqlite3.connect(self.db_path, timeout=timeout)
            conn.row_factory = sqlite3.Row  # 使结果可以通过列名访问
            # 启用WAL模式以提高并发性能
            conn.execute("PRAGMA journal_mode=WAL")
            # 设置繁忙超时，避免长时间等待
            conn.execute("PRAGMA busy_timeout=3000")  # 3秒
            # 启用内存缓存以提高查询速度
            conn.execute("PRAGMA cache_size=-10000")  # 大约10MB缓存
            return conn
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e):
                print(f"[WARNING] 数据库被锁定，超时时间: {timeout}秒")
            raise
    
    def init_database(self):
        """初始化数据库表结构"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # 创建用户表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                email TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                is_active INTEGER DEFAULT 1
            )
        """)
        
        # 创建生成历史表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS generation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                prompt TEXT NOT NULL,
                threshold REAL NOT NULL,
                consistency_score REAL NOT NULL,
                is_consistent INTEGER NOT NULL,
                image_path TEXT,
                result_data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)
        
        # 创建索引以提高查询性能
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_id ON generation_history(user_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_created_at ON generation_history(created_at)
        """)
        
        conn.commit()
        conn.close()
        
        # 创建默认管理员账户（如果不存在）
        self.create_default_admin()
    
    def hash_password(self, password: str) -> str:
        """
        对密码进行哈希处理
        
        Args:
            password: 原始密码
            
        Returns:
            哈希后的密码
        """
        return hashlib.sha256(password.encode()).hexdigest()
    
    def create_default_admin(self):
        """创建默认管理员账户"""
        default_username = "admin"
        default_password = "admin123"
        
        if not self.user_exists(default_username):
            self.register_user(default_username, default_password, "admin@example.com")
            print(f"已创建默认管理员账户: {default_username} / {default_password}")
    
    def register_user(self, username: str, password: str, email: str = "") -> Tuple[bool, str]:
        """
        注册新用户
        
        Args:
            username: 用户名
            password: 密码
            email: 邮箱（可选）
            
        Returns:
            (是否成功, 错误消息)
        """
        import time
        max_retries = 3
        retry_delay = 0.1
        
        for attempt in range(max_retries):
            try:
                conn = self.get_connection()
                cursor = conn.cursor()
                
                password_hash = self.hash_password(password)
                
                cursor.execute("""
                    INSERT INTO users (username, password_hash, email)
                    VALUES (?, ?, ?)
                """, (username, password_hash, email))
                
                conn.commit()
                conn.close()
                return True, "注册成功"
            except sqlite3.IntegrityError:
                conn.close()
                return False, "用户名或邮箱已存在"
            except sqlite3.OperationalError as e:
                conn.close()
                if "database is locked" in str(e).lower() and attempt < max_retries - 1:
                    # 数据库被锁定，等待后重试
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                else:
                    print(f"注册用户时出错 (尝试 {attempt + 1}/{max_retries}): {e}")
                    return False, f"数据库错误: {str(e)}"
            except Exception as e:
                if 'conn' in locals():
                    conn.close()
                print(f"注册用户时出错: {e}")
                return False, f"注册失败: {str(e)}"
        
        return False, "注册失败: 数据库繁忙，请稍后重试"
    
    def user_exists(self, username: str) -> bool:
        """
        检查用户是否存在
        
        Args:
            username: 用户名
            
        Returns:
            用户是否存在
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM users WHERE username = ?", (username,))
            count = cursor.fetchone()[0]
            
            conn.close()
            return count > 0
        except Exception as e:
            print(f"检查用户是否存在时出错: {e}")
            if 'conn' in locals():
                conn.close()
            return False
    
    def verify_user(self, username_or_email: str, password: str) -> Optional[int]:
        """
        验证用户登录（支持用户名或邮箱登录）
        优化版本：减少数据库操作，提高响应速度
        
        Args:
            username_or_email: 用户名或邮箱
            password: 密码
            
        Returns:
            用户ID（如果验证成功），否则返回None
        """
        # 快速验证参数
        if not username_or_email or not password:
            return None
            
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            password_hash = self.hash_password(password)
            
            # 先尝试用户名登录（更常见的情况，提高性能）
            cursor.execute("""
                SELECT id, is_active FROM users 
                WHERE username = ? AND password_hash = ?
            """, (username_or_email, password_hash))
            
            result = cursor.fetchone()
            
            # 如果用户名登录失败，再尝试邮箱登录
            if not result:
                cursor.execute("""
                    SELECT id, is_active FROM users 
                    WHERE email = ? AND password_hash = ?
                """, (username_or_email, password_hash))
                result = cursor.fetchone()
            
            if result and result['is_active']:
                user_id = result['id']
                # 异步更新最后登录时间，不阻塞主流程
                threading.Thread(target=self._update_last_login, args=(user_id,)).start()
                conn.close()
                return user_id
            
            conn.close()
            return None
        except Exception as e:
            print(f"验证用户时出错: {e}")
            if 'conn' in locals():
                conn.close()
            return None
            
    def _update_last_login(self, user_id: int):
        """
        异步更新用户最后登录时间
        
        Args:
            user_id: 用户ID
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE users SET last_login = ? WHERE id = ?
            """, (datetime.now().isoformat(), user_id))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"更新登录时间时出错: {e}")
            if 'conn' in locals():
                conn.close()
    
    def save_generation(self, user_id: int, prompt: str, threshold: float, 
                       consistency_score: float, is_consistent: bool, 
                       image_path: str = "", result_data: dict = None) -> int:
        """
        保存生成历史
        
        Args:
            user_id: 用户ID
            prompt: 提示词
            threshold: 阈值
            consistency_score: 一致性分数
            is_consistent: 是否一致
            image_path: 图像路径
            result_data: 结果数据字典
            
        Returns:
            生成记录的ID
        """
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            result_json = json.dumps(result_data, ensure_ascii=False) if result_data else ""
            
            cursor.execute("""
                INSERT INTO generation_history 
                (user_id, prompt, threshold, consistency_score, is_consistent, image_path, result_data)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (user_id, prompt, threshold, consistency_score, 
                  int(is_consistent), image_path, result_json))
            
            record_id = cursor.lastrowid
            conn.commit()
            
            return record_id
        except Exception as e:
            print(f"[ERROR] 保存生成历史失败: {e}")
            raise
        finally:
            if conn:
                try:
                    conn.close()
                except Exception as e:
                    print(f"[WARNING] 关闭数据库连接失败: {e}")
    
    def get_user_history(self, user_id: int, limit: int = 50) -> List[Dict]:
        """
        获取用户的生成历史
        
        Args:
            user_id: 用户ID
            limit: 返回记录数限制
            
        Returns:
            生成历史列表
        """
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, prompt, threshold, consistency_score, is_consistent, 
                       image_path, result_data, created_at
                FROM generation_history
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (user_id, limit))
            
            rows = cursor.fetchall()
            
            history = []
            for row in rows:
                history.append({
                    'id': row['id'],
                    'prompt': row['prompt'],
                    'threshold': row['threshold'],
                    'consistency_score': row['consistency_score'],
                    'is_consistent': bool(row['is_consistent']),
                    'image_path': row['image_path'],
                    'result_data': row.get('result_data', ''),
                    'created_at': row['created_at']
                })
            
            return history
        except Exception as e:
            print(f"[ERROR] 获取用户历史记录失败: {e}")
            return []
        finally:
            if conn:
                try:
                    conn.close()
                except Exception as e:
                    print(f"[WARNING] 关闭数据库连接失败: {e}")
    
    def get_username_by_id(self, user_id: int) -> Optional[str]:
        """
        根据用户ID获取用户名
        
        Args:
            user_id: 用户ID
            
        Returns:
            用户名，如果用户不存在则返回None
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT username FROM users
                WHERE id = ? AND is_active = 1
            """, (user_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            return row['username'] if row else None
        except Exception as e:
            print(f"获取用户名时出错: {e}")
            if 'conn' in locals():
                conn.close()
            return None
    
    def get_user_info(self, user_id: int) -> Optional[Dict]:
        """
        获取用户信息
        
        Args:
            user_id: 用户ID
            
        Returns:
            用户信息字典，如果用户不存在则返回None
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, username, email, created_at, last_login
            FROM users
            WHERE id = ? AND is_active = 1
        """, (user_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'id': row['id'],
                'username': row['username'],
                'email': row['email'],
                'created_at': row['created_at'],
                'last_login': row['last_login']
            }
        
        return None
    
    def get_statistics(self, user_id: int) -> Dict:
        """
        获取用户统计信息
        
        Args:
            user_id: 用户ID
            
        Returns:
            统计信息字典
        """
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # 总生成次数
            cursor.execute("""
                SELECT COUNT(*) FROM generation_history WHERE user_id = ?
            """, (user_id,))
            total_generations = cursor.fetchone()[0]
            
            # 今日生成次数
            cursor.execute("""
                SELECT COUNT(*) FROM generation_history 
                WHERE user_id = ? AND DATE(created_at) = DATE('now')
            """, (user_id,))
            today_generations = cursor.fetchone()[0]
            
            # 一致性通过次数
            cursor.execute("""
                SELECT COUNT(*) FROM generation_history 
                WHERE user_id = ? AND is_consistent = 1
            """, (user_id,))
            consistent_count = cursor.fetchone()[0]
            
            # 平均分数
            cursor.execute("""
                SELECT AVG(consistency_score) FROM generation_history WHERE user_id = ?
            """, (user_id,))
            avg_score = cursor.fetchone()[0] or 0.0
            
            return {
                'total_generations': total_generations,
                'today_generations': today_generations,
                'consistent_count': consistent_count,
                'inconsistent_count': total_generations - consistent_count,
                'consistency_rate': (consistent_count / total_generations * 100) if total_generations > 0 else 0,
                'average_score': float(avg_score)
            }
        except Exception as e:
            print(f"[ERROR] 获取统计信息失败: {e}")
            return {
                'total_generations': 0,
                'today_generations': 0,
                'consistent_count': 0,
                'inconsistent_count': 0,
                'consistency_rate': 0,
                'average_score': 0.0
            }
        finally:
            if conn:
                try:
                    conn.close()
                except Exception as e:
                    print(f"[WARNING] 关闭数据库连接失败: {e}")
    
    def get_hourly_statistics(self, user_id: int, hours: int = 12) -> List[Dict]:
        """
        获取指定小时数内的统计数据（按小时分组）
        
        Args:
            user_id: 用户ID
            hours: 小时数（默认12小时）
            
        Returns:
            每小时统计列表
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                strftime('%Y-%m-%d %H:00', created_at) as hour,
                COUNT(*) as count
            FROM generation_history
            WHERE user_id = ? 
                AND datetime(created_at) >= datetime('now', '-' || ? || ' hours')
            GROUP BY hour
            ORDER BY hour
        """, (user_id, hours))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'hour': row['hour'],
                'count': row['count']
            })
        
        conn.close()
        return results
    
    def get_model_statistics(self, user_id: int) -> Dict[str, int]:
        """
        获取模型使用统计
        
        Args:
            user_id: 用户ID
            
        Returns:
            模型使用统计字典
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT result_data FROM generation_history WHERE user_id = ?
        """, (user_id,))
        
        model_counts = {}
        for row in cursor.fetchall():
            try:
                result_data = json.loads(row['result_data']) if row['result_data'] else {}
                model_name = result_data.get('model_name', 'unknown')
                if model_name.startswith('lora:'):
                    model_name = 'LoRA模型'
                elif model_name == 'itsc-gan-fusion':
                    model_name = 'ITSC-GAN融合模型'
                elif model_name == 'runwayml/stable-diffusion-v1-5':
                    model_name = '基础模型'
                model_counts[model_name] = model_counts.get(model_name, 0) + 1
            except:
                model_counts['未知模型'] = model_counts.get('未知模型', 0) + 1
        
        conn.close()
        return model_counts
    
    def get_prompt_keywords(self, user_id: int, limit: int = 50) -> List[str]:
        """
        获取提示词中的关键词（用于词云）
        
        Args:
            user_id: 用户ID
            limit: 返回的记录数限制
            
        Returns:
            关键词列表
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT prompt FROM generation_history 
            WHERE user_id = ? 
            ORDER BY created_at DESC 
            LIMIT ?
        """, (user_id, limit))
        
        keywords = []
        for row in cursor.fetchall():
            prompt = row['prompt']
            # 简单提取中文词（可以后续优化使用jieba等）
            import re
            chinese_words = re.findall(r'[\u4e00-\u9fff]+', prompt)
            keywords.extend(chinese_words)
        
        conn.close()
        return keywords

