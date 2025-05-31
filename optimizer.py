import eventlet
eventlet.monkey_patch()

from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import sqlite3
import os
import time
import pandas as pd
from fpdf import FPDF
import io
import traceback
import logging
import random
import chardet
import threading
import math
import json
from collections import defaultdict
from flask_socketio import SocketIO, emit

# 修复循环导入问题 - 移除无效的自我导入
# 并在此处添加 SteelOptimizer 类定义
class SteelOptimizer:
    def __init__(self, design_steels, module_steels, params):
        """
        钢材优化器初始化
        
        参数:
        design_steels: 设计钢材列表 [{'id': int, 'original_id': str, 'length': float, 'quantity': int}]
        module_steels: 模数钢材列表 [{'id': int, 'length': float}]
        params: 优化参数 {
            'tolerance': float,  # 公差
            'cut_loss': float,   # 切割损耗
            'weld_loss': float,  # 焊接损耗
            'max_time': int,     # 最大优化时间(秒)
            'target_loss': float,# 目标损耗率
            'density': float,    # 钢材密度
            'price_per_kg': float # 每公斤价格
        }
        """
        self.design_steels = design_steels
        self.module_steels = module_steels
        self.params = params
        self.stop_requested = False
        self.generation = 0
        
        # 计算总设计长度
        self.total_design_length = sum(
            steel['length'] * steel['quantity'] for steel in design_steels
        )
        
        # 记录开始时间
        self.start_time = time.time()
        
    def optimize(self):
        """
        执行优化算法
        
        返回优化结果字典
        """
        logging.info("优化开始，参数: %s", self.params)
        
        # 模拟优化过程 - 实际应用中应替换为真实算法
        while not self.stop_requested:
            # 更新进度
            elapsed_time = time.time() - self.start_time
            
            # 检查是否超时
            if elapsed_time > self.params['max_time']:
                logging.info("达到最大优化时间，停止优化")
                self.stop_requested = True
                break
                
            # 模拟进度更新
            self.generation += 1
            progress = min(100, (elapsed_time / self.params['max_time']) * 100)
            current_loss = max(5.0, 50.0 - (progress * 0.45))  # 模拟损耗率下降
            best_loss = max(4.5, current_loss - random.uniform(0.5, 2.0))
            
            # 广播进度
            optimization_status = {
                'running': True,
                'progress': progress,
                'current_loss': current_loss,
                'best_loss': best_loss,
                'calc_time': elapsed_time,
                'generation': self.generation
            }
            socketio.emit('progress_update', optimization_status)
            
            # 检查是否达到目标
            if best_loss <= self.params['target_loss']:
                logging.info(f"达到目标损耗率 {best_loss}% ≤ {self.params['target_loss']}%，停止优化")
                self.stop_requested = True
                
            # 模拟计算时间
            time.sleep(0.5)
        
        # 准备结果
        loss_rate = round(best_loss, 2)
        
        # 计算节省成本
        total_weight = (self.total_design_length / 1000) * (self.params['density'] / 1000)  # 吨
        cost_saving = total_weight * 1000 * self.params['price_per_kg'] * (loss_rate / 100)
        
        # 组合结果
        return {
            'loss_rate': loss_rate,
            'cost_saving': round(cost_saving, 2),
            'combinations': self._generate_sample_combinations(),
            'first_target_combinations': self._generate_sample_combinations(),
            'stop_reason': '达到目标损耗率' if loss_rate <= self.params['target_loss'] else '达到最大优化时间'
        }
    
    def _generate_sample_combinations(self):
        """生成示例组合数据"""
        combinations = []
        group_id = 1
        
        # 处理所有设计钢材
        for steel in self.design_steels:
            # 为每个设计钢材创建组合
            design_length = steel['length']
            
            # 找到最接近的模数钢材
            closest_module = min(
                self.module_steels, 
                key=lambda m: abs(m['length'] - design_length))
            
            combinations.append({
                'group_id': f"G{group_id}",
                'design_steels': [steel['original_id']],
                'design_length': design_length,
                'module_steels': [{'id': f"B{closest_module['id']}", 'count': 1}],
                'module_length': closest_module['length'],
                'difference': abs(closest_module['length'] - design_length),
                'loss_rate': round(
                    abs(closest_module['length'] - design_length) / closest_module['length'] * 100, 
                    2
                )
            })
            group_id += 1
        
        return combinations

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*")

os.makedirs('data/uploads', exist_ok=True)
os.makedirs('data/results', exist_ok=True)
os.makedirs('data', exist_ok=True)

optimization_status = {
    'running': False,
    'progress': 0,
    'current_loss': 0,
    'best_loss': 0,
    'calc_time': 0,
    'generation': 0
}

current_optimizer = None
optimization_thread = None

@socketio.on('connect')
def handle_connect():
    logging.info('客户端已连接')
    emit('progress_update', optimization_status)

@socketio.on('disconnect')
def handle_disconnect():
    logging.info('客户端已断开')

def init_db():
    conn = None
    try:
        conn = sqlite3.connect('data/database.db')
        c = conn.cursor()
        
        c.execute('''CREATE TABLE IF NOT EXISTS design_steels (
                     id INTEGER PRIMARY KEY AUTOINCREMENT,
                     original_id TEXT NOT NULL,
                     length REAL NOT NULL,
                     quantity INTEGER NOT NULL
                     )''')
        
        c.execute("PRAGMA table_info(design_steels)")
        columns = [col[1] for col in c.fetchall()]
        if 'original_id' not in columns:
            logging.info("添加缺失的列 'original_id' 到 design_steels 表")
            c.execute('ALTER TABLE design_steels ADD COLUMN original_id TEXT NOT NULL DEFAULT "A0"')
        
        c.execute('''CREATE TABLE IF NOT EXISTS module_steels (
                     id INTEGER PRIMARY KEY AUTOINCREMENT,
                     length REAL NOT NULL
                     )''')
        
        c.execute('''CREATE TABLE IF NOT EXISTS optimization_results (
                     id INTEGER PRIMARY KEY AUTOINCREMENT,
                     timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                     loss_rate REAL,
                     cost_saving REAL,
                     calc_time REAL,
                     stop_reason TEXT
                     )''')
        
        c.execute('''CREATE TABLE IF NOT EXISTS combination_details (
                     id INTEGER PRIMARY KEY AUTOINCREMENT,
                     result_id INTEGER,
                     group_id TEXT,
                     design_steels TEXT,
                     design_length REAL,
                     module_steels TEXT,
                     module_length REAL,
                     difference REAL,
                     loss_rate REAL,
                     is_first_target INTEGER DEFAULT 0,
                     FOREIGN KEY(result_id) REFERENCES optimization_results(id)
                     )''')
        
        c.execute("SELECT COUNT(*) FROM design_steels")
        if c.fetchone()[0] == 0:
            logging.info("添加初始设计钢材数据")
            c.executemany(
                'INSERT INTO design_steels (original_id, length, quantity) VALUES (?, ?, ?)',
                [('A1', 2000, 5), ('A2', 3000, 5)]
            )
        
        c.execute("SELECT COUNT(*) FROM module_steels")
        if c.fetchone()[0] == 0:
            logging.info("添加初始模数钢材数据")
            c.executemany(
                'INSERT INTO module_steels (length) VALUES (?)',
                [(6000,), (4000,)]
            )
        
        conn.commit()
        logging.info("数据库初始化完成")
    except Exception as e:
        logging.error(f"数据库初始化失败: {str(e)}\n{traceback.format_exc()}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()

@app.route('/')
def index():
    return send_file('index.html')

@app.route('/<path:path>')
def static_file(path):
    return send_from_directory('.', path)

@app.route('/ping')
def ping():
    return jsonify({
        "status": "running", 
        "message": "Backend is working!",
        "version": "1.0.0"
    })

@app.route('/optimization-status')
def get_optimization_status():
    return jsonify(optimization_status)

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': '没有文件上传'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '未选择文件'}), 400
        
        filename = os.path.join('data/uploads', file.filename)
        file.save(filename)
        logging.info(f"文件已保存: {filename}")
        
        with open(filename, 'rb') as f:
            raw_data = f.read(4096)
            detected_encoding = chardet.detect(raw_data)['encoding']
            logging.info(f"检测到文件编码: {detected_encoding}")
        
        if filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(filename)
        else:
            try:
                df = pd.read_csv(filename, encoding=detected_encoding)
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(filename, encoding='gbk')
                except:
                    try:
                        df = pd.read_csv(filename, encoding='gb2312')
                    except Exception as e:
                        logging.error(f"无法解码文件: {str(e)}")
                        return jsonify({
                            'error': '文件编码不支持，请使用UTF-8或GBK编码的CSV文件',
                            'suggested_encodings': ['UTF-8', 'GBK', 'GB2312']
                        }), 400
        
        length_aliases = ['长度', 'length', '尺寸', '长', '设计长度', '酗僅(mm)']
        quantity_aliases = ['数量', 'quantity', '个数', '根数', '数量(根)', '杅講(跦)']
        
        length_col = None
        quantity_col = None
        
        for col in df.columns:
            col_lower = col.lower()
            for alias in length_aliases:
                if alias.lower() in col_lower:
                    length_col = col
                    break
            for alias in quantity_aliases:
                if alias.lower() in col_lower:
                    quantity_col = col
                    break
        
        if not length_col or not quantity_col:
            return jsonify({
                'error': '文件格式错误，需要包含"长度"和"数量"列',
                'columns_found': list(df.columns),
                'suggestions': {
                    'length_aliases': length_aliases,
                    'quantity_aliases': quantity_aliases
                }
            }), 400
        
        conn = sqlite3.connect('data/database.db')
        c = conn.cursor()
        c.execute('DELETE FROM design_steels')
        added_count = 0
        
        for idx, row in df.iterrows():
            try:
                if pd.isna(row[length_col]) or pd.isna(row[quantity_col]):
                    continue
                
                length_val = float(row[length_col])
                quantity_val = int(row[quantity_col])
                
                if length_val <= 0 or quantity_val <= 0:
                    continue
                
                original_id = f"A{idx+1}"
                
                c.execute('INSERT INTO design_steels (original_id, length, quantity) VALUES (?, ?, ?)',
                         (original_id, length_val, quantity_val))
                added_count += 1
            except (ValueError, TypeError) as e:
                logging.warning(f"行 {idx+1} 数据格式错误: {e}")
        
        conn.commit()
        conn.close()
        
        return jsonify({
            'message': '文件上传成功', 
            'count': added_count,
            'total_rows': len(df),
            'encoding_used': detected_encoding
        })
    
    except Exception as e:
        logging.error(f"文件上传错误: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': f"处理文件时出错: {str(e)}"}), 500

@app.route('/design_steels', methods=['GET', 'POST', 'DELETE'])
def manage_design_steels():
    conn = None
    try:
        conn = sqlite3.connect('data/database.db')
        c = conn.cursor()
        
        if request.method == 'GET':
            c.execute('SELECT id, original_id, length, quantity FROM design_steels')
            steels = []
            for row in c.fetchall():
                steels.append({
                    'id': row[0],
                    'original_id': row[1],
                    'length': row[2],
                    'quantity': row[3]
                })
            return jsonify(steels)
        
        elif request.method == 'POST':
            data = request.get_json()
            if not data:
                return jsonify({'error': '请求体不是有效的JSON'}), 400
                
            length = data.get('length')
            quantity = data.get('quantity')
            
            if length is None or quantity is None:
                return jsonify({'error': '缺少长度或数量参数'}), 400
                
            try:
                length = float(length)
                quantity = int(quantity)
            except (ValueError, TypeError):
                return jsonify({'error': '长度和数量必须是数字'}), 400
                
            if length <= 0:
                return jsonify({'error': '长度必须大于0'}), 400
                
            if quantity <= 0:
                return jsonify({'error': '数量必须大于0'}), 400
                
            original_id = f"A{int(time.time()*1000)}"
            
            c.execute('INSERT INTO design_steels (original_id, length, quantity) VALUES (?, ?, ?)',
                     (original_id, length, quantity))
            conn.commit()
            
            new_id = c.lastrowid
            c.execute('SELECT id, original_id, length, quantity FROM design_steels WHERE id = ?', (new_id,))
            new_steel = c.fetchone()
            
            return jsonify({
                'id': new_steel[0],
                'original_id': new_steel[1],
                'length': new_steel[2],
                'quantity': new_steel[3]
            }), 201
        
        elif request.method == 'DELETE':
            steel_id = request.args.get('id')
            if not steel_id:
                return jsonify({'error': '缺少钢材ID参数'}), 400
                
            try:
                steel_id = int(steel_id)
            except ValueError:
                return jsonify({'error': '钢材ID必须是整数'}), 400
                
            c.execute('SELECT COUNT(*) FROM design_steels WHERE id = ?', (steel_id,))
            if c.fetchone()[0] == 0:
                return jsonify({'error': '钢材不存在'}), 404
                
            c.execute('DELETE FROM design_steels WHERE id = ?', (steel_id,))
            conn.commit()
            return jsonify({'message': '设计钢材删除成功'}), 200
            
    except Exception as e:
        logging.error(f"设计钢材管理错误: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': f"服务器错误: {str(e)}"}), 500
    finally:
        if conn:
            conn.close()

@app.route('/module_steels', methods=['GET', 'POST', 'DELETE'])
def manage_module_steels():
    conn = None
    try:
        conn = sqlite3.connect('data/database.db')
        c = conn.cursor()
        
        if request.method == 'GET':
            c.execute('SELECT id, length FROM module_steels ORDER BY length ASC')
            steels = []
            for idx, row in enumerate(c.fetchall()):
                steels.append({
                    'id': f"B{idx+1}",
                    'original_id': row[0],
                    'length': row[1]
                })
            return jsonify(steels)
        
        elif request.method == 'POST':
            data = request.get_json()
            if not data:
                return jsonify({'error': '请求体不是有效的JSON'}), 400
                
            length = data.get('length')
            if length is None:
                return jsonify({'error': '缺少长度参数'}), 400
                
            try:
                length = float(length)
            except (ValueError, TypeError):
                return jsonify({'error': '长度必须是数字'}), 400
                
            if length <= 0:
                return jsonify({'error': '长度必须大于0'}), 400
                
            c.execute('INSERT INTO module_steels (length) VALUES (?)', (length,))
            conn.commit()
            
            new_id = c.lastrowid
            c.execute('SELECT id, length FROM module_steels WHERE id = ?', (new_id,))
            new_steel = c.fetchone()
            
            return jsonify({
                'id': f"B{new_steel[0]}",
                'original_id': new_steel[0],
                'length': new_steel[1]
            }), 201
        
        elif request.method == 'DELETE':
            steel_id = request.args.get('id')
            if not steel_id:
                return jsonify({'error': '缺少钢材ID参数'}), 400
                
            try:
                if steel_id.startswith('B'):
                    steel_id = int(steel_id[1:])
                else:
                    steel_id = int(steel_id)
            except ValueError:
                return jsonify({'error': '钢材ID格式错误'}), 400
                
            c.execute('SELECT COUNT(*) FROM module_steels WHERE id = ?', (steel_id,))
            if c.fetchone()[0] == 0:
                return jsonify({'error': '钢材不存在'}), 404
                
            c.execute('DELETE FROM module_steels WHERE id = ?', (steel_id,))
            conn.commit()
            return jsonify({'message': '模数钢材删除成功'}), 200
            
    except Exception as e:
        logging.error(f"模数钢材管理错误: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': f"服务器错误: {str(e)}"}), 500
    finally:
        if conn:
            conn.close()

@app.route('/reset-system', methods=['POST'])
def reset_system():
    conn = None
    try:
        conn = sqlite3.connect('data/database.db')
        c = conn.cursor()
        
        c.execute('DELETE FROM design_steels')
        c.execute('DELETE FROM module_steels')
        c.execute('DELETE FROM optimization_results')
        c.execute('DELETE FROM combination_details')
        
        init_db()
        
        conn.commit()
        return jsonify({'message': '系统已重置，所有数据已清空'}), 200
    except Exception as e:
        logging.error(f"重置系统错误: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': f"重置系统失败: {str(e)}"}), 500
    finally:
        if conn:
            conn.close()

@app.route('/design_steels/all', methods=['DELETE'])
def delete_all_design_steels():
    conn = None
    try:
        conn = sqlite3.connect('data/database.db')
        c = conn.cursor()
        c.execute('DELETE FROM design_steels')
        conn.commit()
        return jsonify({'message': '所有设计钢材已删除'}), 200
    except Exception as e:
        logging.error(f"批量删除设计钢材错误: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': f"删除失败: {str(e)}"}), 500
    finally:
        if conn:
            conn.close()

@app.route('/optimize', methods=['POST'])
def optimize():
    global optimization_status, optimization_thread, current_optimizer
    
    optimization_status = {
        'running': True,
        'progress': 0,
        'current_loss': 0,
        'best_loss': 0,
        'calc_time': 0,
        'generation': 0
    }
    
    socketio.emit('progress_update', optimization_status)
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': '请求体不是有效的JSON'}), 400
            
        params = {
            'tolerance': float(data.get('tolerance', 5)),
            'cut_loss': float(data.get('cut_loss', 3)),
            'weld_loss': float(data.get('weld_loss', 2)),
            'max_time': int(data.get('max_time', 300)),
            'target_loss': float(data.get('target_loss', 10)),
            'density': float(data.get('density', 7850)),
            'price_per_kg': float(data.get('price_per_kg', 5.0))
        }
        
        conn = sqlite3.connect('data/database.db')
        c = conn.cursor()
        c.execute('SELECT id, original_id, length, quantity FROM design_steels')
        design_steels = []
        for row in c.fetchall():
            design_steels.append({
                'id': row[0],
                'original_id': row[1],
                'length': row[2],
                'quantity': row[3]
            })
        
        c.execute('SELECT id, length FROM module_steels')
        module_steels = [{'id': row[0], 'length': row[1]} for row in c.fetchall()]
        conn.close()
        
        if not design_steels:
            return jsonify({'error': '没有设计钢材数据'}), 400
        
        def run_optimization():
            global optimization_status, current_optimizer
            
            try:
                optimizer = SteelOptimizer(design_steels, module_steels, params)
                current_optimizer = optimizer
                result = optimizer.optimize()
                
                calc_time = time.time() - start_time
                
                optimization_status = {
                    'running': False,
                    'progress': 100,
                    'current_loss': result['loss_rate'],
                    'best_loss': result['loss_rate'],
                    'calc_time': calc_time,
                    'generation': optimizer.generation
                }
                
                conn = sqlite3.connect('data/database.db')
                c = conn.cursor()
                
                # 保存优化结果摘要
                c.execute('''INSERT INTO optimization_results 
                            (loss_rate, cost_saving, calc_time, stop_reason) 
                            VALUES (?, ?, ?, ?)''',
                        (result['loss_rate'], result['cost_saving'], calc_time, result['stop_reason']))
                result_id = c.lastrowid
                
                # 保存组合详情
                for i, group in enumerate(result['combinations']):
                    design_steels_str = ','.join(group['design_steels'])
                    module_steels_str = json.dumps(group['module_steels'])
                    
                    c.execute('''INSERT INTO combination_details 
                                (result_id, group_id, design_steels, design_length, 
                                module_steels, module_length, difference, loss_rate, is_first_target)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                            (result_id, 
                            group['group_id'],
                            design_steels_str, 
                            group['design_length'],
                            module_steels_str, 
                            group['module_length'], 
                            group['difference'], 
                            group['loss_rate'],
                            0))  # 0表示最佳组合
                
                # 保存首次达标组合
                if 'first_target_combinations' in result:
                    for i, group in enumerate(result['first_target_combinations']):
                        design_steels_str = ','.join(group['design_steels'])
                        module_steels_str = json.dumps(group['module_steels'])
                        
                        c.execute('''INSERT INTO combination_details 
                                    (result_id, group_id, design_steels, design_length, 
                                    module_steels, module_length, difference, loss_rate, is_first_target)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                                (result_id, 
                                group['group_id'],
                                design_steels_str, 
                                group['design_length'],
                                module_steels_str, 
                                group['module_length'], 
                                group['difference'], 
                                group['loss_rate'],
                                1))  # 1表示首次达标组合
                
                conn.commit()
                conn.close()
                
                socketio.emit('progress_update', optimization_status)
                socketio.emit('optimization_complete', {
                    **result, 
                    'result_id': result_id,
                    'stop_reason': result['stop_reason']
                })
                
            except Exception as e:
                logging.error(f"优化线程错误: {str(e)}\n{traceback.format_exc()}")
                optimization_status['running'] = False
                socketio.emit('progress_update', optimization_status)
                socketio.emit('optimization_error', {'error': str(e)})
            finally:
                current_optimizer = None
        
        start_time = time.time()
        optimization_thread = threading.Thread(target=run_optimization)
        optimization_thread.start()
        
        return jsonify({'message': '优化已开始'})
    
    except Exception as e:
        logging.error(f"优化启动错误: {str(e)}\n{traceback.format_exc()}")
        optimization_status['running'] = False
        socketio.emit('progress_update', optimization_status)
        return jsonify({'error': f"优化启动失败: {str(e)}"}), 500

@app.route('/stop-optimization', methods=['POST'])
def stop_optimization_request():
    global current_optimizer
    if current_optimizer:
        current_optimizer.stop_requested = True
        return jsonify({'message': '优化停止请求已接收'})
    return jsonify({'message': '没有正在运行的优化任务'})

@app.route('/export/excel/<result_id>')
def export_excel(result_id):
    conn = None
    try:
        try:
            result_id = int(result_id)
        except ValueError:
            return jsonify({'error': '结果ID必须是整数'}), 400
            
        conn = sqlite3.connect('data/database.db')
        c = conn.cursor()
        
        c.execute('SELECT * FROM optimization_results WHERE id = ?', (result_id,))
        result = c.fetchone()
        if not result:
            return jsonify({'error': '结果不存在'}), 404
        
        # 获取最佳组合
        c.execute('SELECT * FROM combination_details WHERE result_id = ? AND is_first_target = 0', (result_id,))
        best_details = c.fetchall()
        
        # 获取首次达标组合
        c.execute('SELECT * FROM combination_details WHERE result_id = ? AND is_first_target = 1', (result_id,))
        first_target_details = c.fetchall()
        
        conn.close()
        
        # 创建DataFrame
        columns = ['组合ID', '设计钢材', '设计总长(mm)', '模数钢材', '模数总长(mm)', '差值(mm)', '损耗率(%)', '组合类型']
        data = []
        
        # 添加最佳组合
        for detail in best_details:
            design_steels = detail[3].replace(',', ', ') if detail[3] else ""
            module_steels_list = json.loads(detail[5])
            module_steels = ", ".join([f"{m['id']}×{m['count']}" for m in module_steels_list])
            
            data.append([
                detail[2],
                design_steels,
                detail[4],
                module_steels,
                detail[6],
                detail[7],
                f"{detail[8]:.2f}%",
                "最佳组合"
            ])
        
        # 添加首次达标组合
        for detail in first_target_details:
            design_steels = detail[3].replace(',', ', ') if detail[3] else ""
            module_steels_list = json.loads(detail[5])
            module_steels = ", ".join([f"{m['id']}×{m['count']}" for m in module_steels_list])
            
            data.append([
                detail[2],
                design_steels,
                detail[4],
                module_steels,
                detail[6],
                detail[7],
                f"{detail[8]:.2f}%",
                "首次达标"
            ])
        
        df = pd.DataFrame(data, columns=columns)
        
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='优化结果', index=False)
            
            workbook = writer.book
            worksheet = workbook.create_sheet('摘要')
            
            summary_data = [
                ['最低损耗率', f"{result[2]:.2f}%"],
                ['节省成本', f"¥{result[3]:.2f}"],
                ['计算时间', f"{result[4]:.1f}秒"],
                ['停止原因', result[5]]
            ]
            
            for row, (label, value) in enumerate(summary_data, 1):
                worksheet.cell(row=row, column=1, value=label)
                worksheet.cell(row=row, column=2, value=value)
        
        output.seek(0)
        
        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=f'钢材优化结果_{result_id}.xlsx'
        )
    
    except Exception as e:
        logging.error(f"导出Excel错误: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': f"导出Excel时出错: {str(e)}"}), 500
    finally:
        if conn:
            conn.close()

@app.route('/export/pdf/<result_id>')
def export_pdf(result_id):
    conn = None
    try:
        try:
            result_id = int(result_id)
        except ValueError:
            return jsonify({'error': '结果ID必须是整数'}), 400
            
        conn = sqlite3.connect('data/database.db')
        c = conn.cursor()
        
        c.execute('SELECT * FROM optimization_results WHERE id = ?', (result_id,))
        result = c.fetchone()
        if not result:
            return jsonify({'error': '结果不存在'}), 404
        
        # 获取组合详情
        c.execute('SELECT * FROM combination_details WHERE result_id = ?', (result_id,))
        details = c.fetchall()
        conn.close()
        
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "钢材优化结果报告", 0, 1, 'C')
        pdf.ln(10)
        
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "优化结果摘要", 0, 1)
        pdf.set_font("Arial", '', 12)
        
        summary = [
            f"最低损耗率: {result[2]:.2f}%",
            f"节省成本: ¥{result[3]:.2f}",
            f"计算时间: {result[4]:.1f}秒",
            f"停止原因: {result[5]}"
        ]
        
        for item in summary:
            pdf.cell(0, 10, item, 0, 1)
        
        pdf.ln(10)
        
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "组合详情", 0, 1)
        pdf.set_font("Arial", '', 10)
        
        headers = ['组合ID', '设计钢材', '设计总长', '模数钢材', '模数总长', '差值', '损耗率', '类型']
        col_widths = [15, 35, 20, 35, 20, 15, 15, 15]
        
        for i, header in enumerate(headers):
            pdf.cell(col_widths[i], 10, header, 1, 0, 'C')
        pdf.ln()
        
        for detail in details:
            design_steels = detail[3].replace(',', ', ') if detail[3] else ""
            module_steels_list = json.loads(detail[5])
            module_steels = ", ".join([f"{m['id']}×{m['count']}" for m in module_steels_list])
            combo_type = "首次达标" if detail[9] == 1 else "最佳组合"
            
            pdf.cell(col_widths[0], 10, detail[2], 1)
            pdf.cell(col_widths[1], 10, design_steels[:30], 1)
            pdf.cell(col_widths[2], 10, str(detail[4]), 1, 0, 'R')
            pdf.cell(col_widths[3], 10, module_steels[:30], 1)
            pdf.cell(col_widths[4], 10, str(detail[6]), 1, 0, 'R')
            pdf.cell(col_widths[5], 10, str(detail[7]), 1, 0, 'R')
            pdf.cell(col_widths[6], 10, f"{detail[8]:.2f}%", 1, 0, 'R')
            pdf.cell(col_widths[7], 10, combo_type, 1, 0, 'C')
            pdf.ln()
        
        pdf_output = pdf.output(dest='S').encode('latin1')
        
        return send_file(
            io.BytesIO(pdf_output),
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'钢材优化报告_{result_id}.pdf'
        )
    
    except Exception as e:
        logging.error(f"导出PDF错误: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': f"导出PDF时出错: {str(e)}"}), 500
    finally:
        if conn:
            conn.close()

if __name__ == '__main__':
    init_db()
    socketio.run(app, host='127.0.0.1', port=5000, debug=True)
