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
from optimizer import SteelOptimizer  # 添加导入SteelOptimizer

# ======== 修改日志配置为DEBUG级别 ========
logging.basicConfig(
    level=logging.DEBUG,  # 改为 DEBUG 级别获取更多信息
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
# ======== 日志配置结束 ========

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*")  # 初始化SocketIO

# 确保数据目录存在
os.makedirs('data/uploads', exist_ok=True)
os.makedirs('data/results', exist_ok=True)
os.makedirs('data', exist_ok=True)

# 全局优化状态
optimization_status = {
    'running': False,
    'progress': 0,
    'current_loss': 0,
    'best_loss': 0,
    'calc_time': 0,
    'generation': 0
}

# 当前优化器实例
current_optimizer = None

# 优化线程
optimization_thread = None

# SocketIO 事件处理
@socketio.on('connect')
def handle_connect():
    logging.info('客户端已连接')
    emit('progress_update', optimization_status)

@socketio.on('disconnect')
def handle_disconnect():
    logging.info('客户端已断开')

# 数据库初始化
def init_db():
    conn = None
    try:
        conn = sqlite3.connect('data/database.db')
        c = conn.cursor()
        
        # 创建设计钢材表
        c.execute('''CREATE TABLE IF NOT EXISTS design_steels (
                     id INTEGER PRIMARY KEY AUTOINCREMENT,
                     original_id TEXT NOT NULL,
                     length REAL NOT NULL,
                     quantity INTEGER NOT NULL
                     )''')
        
        # 检查表结构并添加缺失的列
        c.execute("PRAGMA table_info(design_steels)")
        columns = [col[1] for col in c.fetchall()]
        if 'original_id' not in columns:
            logging.info("添加缺失的列 'original_id' 到 design_steels 表")
            c.execute('ALTER TABLE design_steels ADD COLUMN original_id TEXT NOT NULL DEFAULT "A0"')
        
        # 创建模数钢材表
        c.execute('''CREATE TABLE IF NOT EXISTS module_steels (
                     id INTEGER PRIMARY KEY AUTOINCREMENT,
                     length REAL NOT NULL
                     )''')
        
        # 创建优化结果表
        c.execute('''CREATE TABLE IF NOT EXISTS optimization_results (
                     id INTEGER PRIMARY KEY AUTOINCREMENT,
                     timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                     loss_rate REAL,
                     cost_saving REAL,
                     calc_time REAL,
                     stop_reason TEXT  -- 确保包含 stop_reason 列
                     )''')
        
        # 检查 optimization_results 表是否有 stop_reason 列
        c.execute("PRAGMA table_info(optimization_results)")
        columns = [col[1] for col in c.fetchall()]
        if 'stop_reason' not in columns:
            logging.info("添加缺失的列 'stop_reason' 到 optimization_results 表")
            c.execute('ALTER TABLE optimization_results ADD COLUMN stop_reason TEXT')
        
        # 创建组合详情表
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
                     FOREIGN KEY(result_id) REFERENCES optimization_results(id)
                     )''')
        
        # 添加初始测试数据
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

# 添加前端路由
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
    """获取优化进度"""
    return jsonify(optimization_status)

@app.route('/upload', methods=['POST'])
def upload_file():
    """处理设计钢材文件上传"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': '没有文件上传'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '未选择文件'}), 400
        
        # 保存文件
        filename = os.path.join('data/uploads', file.filename)
        file.save(filename)
        logging.info(f"文件已保存: {filename}")
        
        # 检测文件编码
        with open(filename, 'rb') as f:
            raw_data = f.read(4096)
            detected_encoding = chardet.detect(raw_data)['encoding']
            logging.info(f"检测到文件编码: {detected_encoding}")
        
        # 处理Excel/CSV文件
        if filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(filename)
        else:  # CSV
            try:
                # 尝试使用检测到的编码
                df = pd.read_csv(filename, encoding=detected_encoding)
            except UnicodeDecodeError:
                # 尝试常见中文编码
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
        
        # 更灵活的列名检查
        length_col = None
        quantity_col = None
        
        # 可能的列名变体
        length_aliases = ['长度', 'length', '尺寸', '长', '设计长度', '酗僅(mm)']
        quantity_aliases = ['数量', 'quantity', '个数', '根数', '数量(根)', '杅講(跦)']
        
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
        
        # 保存到数据库
        conn = sqlite3.connect('data/database.db')
        c = conn.cursor()
        c.execute('DELETE FROM design_steels')
        added_count = 0
        
        for idx, row in df.iterrows():
            try:
                # 跳过空行
                if pd.isna(row[length_col]) or pd.isna(row[quantity_col]):
                    continue
                
                length_val = float(row[length_col])
                quantity_val = int(row[quantity_col])
                
                if length_val <= 0 or quantity_val <= 0:
                    continue
                
                # 生成原始ID (A1, A2等)
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

# 设计钢材管理API
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
                
            # 验证数据
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
                
            # 生成唯一ID
            original_id = f"A{int(time.time()*1000)}"
            
            c.execute('INSERT INTO design_steels (original_id, length, quantity) VALUES (?, ?, ?)',
                     (original_id, length, quantity))
            conn.commit()
            
            # 获取新添加的钢材
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

# 模数钢材管理API
@app.route('/module_steels', methods=['GET', 'POST', 'DELETE'])
def manage_module_steels():
    conn = None
    try:
        conn = sqlite3.connect('data/database.db')
        c = conn.cursor()
        
        if request.method == 'GET':
            # 按长度升序排序并重新编号
            c.execute('SELECT id, length FROM module_steels ORDER BY length ASC')
            steels = []
            for idx, row in enumerate(c.fetchall()):
                steels.append({
                    'id': f"B{idx+1}",  # 重新编号从B1开始
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
            
            # 获取新添加的钢材
            new_id = c.lastrowid
            c.execute('SELECT id, length FROM module_steels WHERE id = ?', (new_id,))
            new_steel = c.fetchone()
            
            return jsonify({
                'id': f"B{new_steel[0]}",  # 返回带字母前缀的ID
                'original_id': new_steel[0],
                'length': new_steel[1]
            }), 201
        
        elif request.method == 'DELETE':
            steel_id = request.args.get('id')
            if not steel_id:
                return jsonify({'error': '缺少钢材ID参数'}), 400
                
            try:
                # 从B1格式中提取数字ID
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

# 添加重置系统功能
@app.route('/reset-system', methods=['POST'])
def reset_system():
    """重置系统，清空所有数据"""
    conn = None
    try:
        conn = sqlite3.connect('data/database.db')
        c = conn.cursor()
        
        # 清空所有表
        c.execute('DELETE FROM design_steels')
        c.execute('DELETE FROM module_steels')
        c.execute('DELETE FROM optimization_results')
        c.execute('DELETE FROM combination_details')
        
        # 重新初始化数据库
        init_db()
        
        conn.commit()
        return jsonify({'message': '系统已重置，所有数据已清空'}), 200
    except Exception as e:
        logging.error(f"重置系统错误: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': f"重置系统失败: {str(e)}"}), 500
    finally:
        if conn:
            conn.close()

# 添加设计钢材批量删除功能
@app.route('/design_steels/all', methods=['DELETE'])
def delete_all_design_steels():
    """删除所有设计钢材"""
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
    """启动优化计算"""
    global optimization_status, optimization_thread, current_optimizer
    
    # 重置优化状态
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
        # 获取参数
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
        
        # 从数据库获取设计钢材
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
        
        # 获取模数钢材
        c.execute('SELECT id, length FROM module_steels')
        module_steels = [{'id': row[0], 'length': row[1]} for row in c.fetchall()]
        conn.close()
        
        if not design_steels:
            return jsonify({'error': '没有设计钢材数据'}), 400
        
        # 在线程中运行优化
        def run_optimization():
            global optimization_status, current_optimizer
            
            try:
                # 创建优化器并运行
                optimizer = SteelOptimizer(design_steels, module_steels, params)
                current_optimizer = optimizer  # 保存当前优化器实例
                result = optimizer.optimize()
                
                # 计算最终结果
                calc_time = time.time() - start_time
                
                # 更新最终状态
                optimization_status = {
                    'running': False,
                    'progress': 100,
                    'current_loss': result['loss_rate'],
                    'best_loss': result['loss_rate'],
                    'calc_time': calc_time,
                    'generation': optimizer.generation
                }
                
                # 保存结果到数据库
                conn = sqlite3.connect('data/database.db')
                c = conn.cursor()
                
                # 保存优化结果摘要
                c.execute('''INSERT INTO optimization_results 
                            (loss_rate, cost_saving, calc_time, stop_reason) 
                            VALUES (?, ?, ?, ?)''',
                        (result['loss_rate'], result['cost_saving'], result['calc_time'], result['stop_reason']))
                result_id = c.lastrowid
                
                # 保存组合详情
                for i, group in enumerate(result['combinations']):
                    # 确保design_steels是字符串
                    design_steels_str = ','.join(group['design_steels'])
                    
                    # 使用JSON存储模数钢材
                    module_steels_str = json.dumps(group['module_steels'])
                    
                    c.execute('''INSERT INTO combination_details 
                                (result_id, group_id, design_steels, design_length, 
                                module_steels, module_length, difference, loss_rate)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                            (result_id, 
                            f"G{i+1}",
                            design_steels_str, 
                            group['design_length'],
                            module_steels_str, 
                            group['module_length'], 
                            group['difference'], 
                            group['loss_rate']))
                
                conn.commit()
                conn.close()
                
                # 发送最终结果
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
                current_optimizer = None  # 清理当前优化器实例
        
        # 启动优化线程
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
    """停止优化计算"""
    global current_optimizer
    if current_optimizer:
        current_optimizer.stop_requested = True
        return jsonify({'message': '优化停止请求已接收'})
    return jsonify({'message': '没有正在运行的优化任务'})

@app.route('/export/excel/<result_id>')
def export_excel(result_id):
    """导出Excel结果"""
    conn = None
    try:
        # 验证result_id
        try:
            result_id = int(result_id)
        except ValueError:
            return jsonify({'error': '结果ID必须是整数'}), 400
            
        # 从数据库获取结果
        conn = sqlite3.connect('data/database.db')
        c = conn.cursor()
        
        # 获取结果摘要
        c.execute('SELECT * FROM optimization_results WHERE id = ?', (result_id,))
        result = c.fetchone()
        if not result:
            return jsonify({'error': '结果不存在'}), 404
        
        # 获取组合详情
        c.execute('SELECT * FROM combination_details WHERE result_id = ?', (result_id,))
        details = c.fetchall()
        conn.close()
        
        # 创建DataFrame
        columns = ['组合ID', '设计钢材', '设计总长(mm)', '模数钢材', '模数总长(mm)', '差值(mm)', '损耗率(%)']
        data = []
        
        for detail in details:
            # 格式化设计钢材显示
            design_steels = detail[3].replace(',', ', ') if detail[3] else ""
            
            # 解析模数钢材JSON
            module_steels_list = json.loads(detail[5])
            module_steels = ", ".join([f"{m['id']}×{m['count']}" for m in module_steels_list])
            
            data.append([
                detail[2],  # group_id
                design_steels,  
                detail[4],  # design_length
                module_steels, 
                detail[6],  # module_length
                detail[7],  # difference
                f"{detail[8]:.2f}%"  # loss_rate
            ])
        
        df = pd.DataFrame(data, columns=columns)
        
        # 创建Excel文件
        output = io.BytesIO()
        
        # 使用openpyxl引擎
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='优化结果', index=False)
            
            # 添加摘要信息
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
        
        # 返回Excel文件
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
    """导出PDF结果"""
    conn = None
    try:
        # 验证result_id
        try:
            result_id = int(result_id)
        except ValueError:
            return jsonify({'error': '结果ID必须是整数'}), 400
            
        # 从数据库获取结果
        conn = sqlite3.connect('data/database.db')
        c = conn.cursor()
        
        # 获取结果摘要
        c.execute('SELECT * FROM optimization_results WHERE id = ?', (result_id,))
        result = c.fetchone()
        if not result:
            return jsonify({'error': '结果不存在'}), 404
        
        # 获取组合详情
        c.execute('SELECT * FROM combination_details WHERE result_id = ?', (result_id,))
        details = c.fetchall()
        conn.close()
        
        # 创建PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "钢材优化结果报告", 0, 1, 'C')
        pdf.ln(10)
        
        # 添加摘要
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
        
        # 添加组合详情
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "组合详情", 0, 1)
        pdf.set_font("Arial", '', 10)
        
        # 表头
        headers = ['组合ID', '设计钢材', '设计总长', '模数钢材', '模数总长', '差值', '损耗率']
        col_widths = [20, 40, 25, 40, 25, 20, 20]
        
        for i, header in enumerate(headers):
            pdf.cell(col_widths[i], 10, header, 1, 0, 'C')
        pdf.ln()
        
        # 表格内容
        for detail in details:
            # 格式化设计钢材显示
            design_steels = detail[3].replace(',', ', ') if detail[3] else ""
            
            # 解析模数钢材JSON
            module_steels_list = json.loads(detail[5])
            module_steels = ", ".join([f"{m['id']}×{m['count']}" for m in module_steels_list])
            
            pdf.cell(col_widths[0], 10, detail[2], 1)  # group_id
            pdf.cell(col_widths[1], 10, design_steels[:35], 1)  # design_steels
            pdf.cell(col_widths[2], 10, str(detail[4]), 1, 0, 'R')  # design_length
            pdf.cell(col_widths[3], 10, module_steels[:35], 1)  # module_steels
            pdf.cell(col_widths[4], 10, str(detail[6]), 1, 0, 'R')  # module_length
            pdf.cell(col_widths[5], 10, str(detail[7]), 1, 0, 'R')  # difference
            pdf.cell(col_widths[6], 10, f"{detail[8]:.2f}%", 1, 0, 'R')  # loss_rate
            pdf.ln()
        
        # 保存到内存
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
