import time
import random
import logging
import traceback
from collections import defaultdict

class SteelOptimizer:
    def __init__(self, design_steels, module_steels, params):
        self.design_steels = self._expand_design_steels(design_steels)
        self.module_steels = module_steels
        self.params = params
        self.best_solution = None
        self.best_loss = float('inf')
        self.start_time = time.time()
        self.generation = 0
        self.no_improvement_count = 0
        self.stop_reason = "优化完成"
        
        # 对模数钢材按长度排序，便于后续处理
        self.module_steels_sorted = sorted(module_steels, key=lambda x: x['length'], reverse=True)
        
        # 创建模数钢材ID到长度的映射
        self.module_length_map = {m['id']: m['length'] for m in module_steels}
        
        # 创建设计钢材ID到长度的映射
        self.design_length_map = {}
        for steel in design_steels:
            for i in range(steel['quantity']):
                steel_id = f"{steel['original_id']}_{i+1}"
                self.design_length_map[steel_id] = steel['length']

    def _expand_design_steels(self, design_steels):
        """将设计钢材展开为单体列表"""
        expanded = []
        for steel in design_steels:
            for i in range(steel['quantity']):
                expanded.append({
                    'id': f"{steel['original_id']}_{i+1}",  # 唯一ID
                    'original_id': steel['original_id'],
                    'length': steel['length']
                })
        return expanded

    def _generate_random_solution(self):
        """生成随机初始解决方案"""
        solution = []
        remaining_steels = self.design_steels.copy()
        random.shuffle(remaining_steels)
        
        # 根据钢材长度分组，相近长度的放在一起
        remaining_steels.sort(key=lambda x: x['length'])
        
        # 创建随机分组
        while remaining_steels:
            # 基于长度确定组大小：长钢材组小，短钢材组大
            avg_length = sum(s['length'] for s in remaining_steels) / len(remaining_steels)
            max_group_size = 5 if avg_length > 2000 else 8
            group_size = min(random.randint(1, max_group_size), len(remaining_steels))
            
            group_steels = remaining_steels[:group_size]
            remaining_steels = remaining_steels[group_size:]
            
            # 分配模数钢材
            module_assignment = self._assign_modules(group_steels)
            solution.append({
                'design_steels': [s['id'] for s in group_steels],
                'design_length': sum(s['length'] for s in group_steels),
                'module_steels': module_assignment
            })
        
        return solution

    def _greedy_combination(self, required_length):
        """贪心算法处理大尺寸钢材"""
        sorted_modules = sorted(self.module_steels, key=lambda x: x['length'], reverse=True)
        combination = []
        remaining = required_length
        
        for module in sorted_modules:
            if module['length'] <= remaining:
                count = remaining // module['length']
                if count > 0:
                    combination.append({
                        'id': module['id'],
                        'length': module['length'],
                        'count': count
                    })
                    remaining -= count * module['length']
        
        if remaining > 0:
            # 添加最小的能覆盖剩余长度的钢材
            closest = min(self.module_steels, key=lambda m: abs(m['length'] - remaining))
            combination.append({'id': closest['id'], 'length': closest['length'], 'count': 1})
        
        return combination

    def _find_best_combination(self, required_length):
        """使用动态规划找到最接近的组合 - 改进算法"""
        tolerance = self.params['tolerance']
        
        # 将长度转换为整数（毫米）
        required_length = int(round(required_length))
        tolerance = int(round(tolerance))
        
        # 添加边界检查
        if required_length <= 0:
            logging.error(f"无效的 required_length: {required_length}")
            # 使用最小的模数钢材
            min_module = min(self.module_steels, key=lambda m: m['length'])
            return [{'id': min_module['id'], 'length': min_module['length'], 'count': 1}]
        
        # 如果所需长度超过20000mm，使用贪心算法
        if required_length > 20000:
            return self._greedy_combination(required_length)
            
        min_length = max(0, required_length - tolerance)
        max_length = required_length + tolerance
        
        # 确保 max_length 不会过大
        if max_length > 50000:  # 设置最大上限
            return self._greedy_combination(required_length)
        
        # 记录转换后的值
        logging.info(f"动态规划参数: required_length={required_length}, "
                     f"min_length={min_length}, max_length={max_length}, "
                     f"tolerance={tolerance}")
        
        try:
            # 初始化DP数组
            dp = [None] * (max_length + 1)
            dp[0] = []
            
            # 对每个模数钢材
            for module in self.module_steels_sorted:
                # 转换为整数毫米
                module_length = int(round(module['length']))
                
                # 跳过无效长度
                if module_length <= 0:
                    logging.warning(f"跳过无效的模数钢材长度: {module}")
                    continue
                    
                # 确保 module_length 在有效范围内
                if module_length >= len(dp):
                    continue
                    
                # 动态规划核心逻辑
                for current_length in range(module_length, len(dp)):
                    if current_length - module_length < 0:
                        continue
                    
                    if dp[current_length - module_length] is not None:
                        new_combination = dp[current_length - module_length] + [module]
                        
                        # 如果当前长度还没有组合，或者新组合的钢材数量更少
                        if dp[current_length] is None or len(new_combination) < len(dp[current_length]):
                            dp[current_length] = new_combination
                        
                        # 如果当前长度在允许范围内，记录
                        if min_length <= current_length <= max_length:
                            # 转换组合格式
                            combination_formatted = []
                            # 统计每个钢材的数量
                            count_dict = defaultdict(int)
                            for m in new_combination:
                                count_dict[m['id']] += 1
                            for m_id, count in count_dict.items():
                                combination_formatted.append({
                                    'id': m_id, 
                                    'length': self.module_length_map[m_id], 
                                    'count': count
                                })
                            return combination_formatted
        except Exception as e:
            logging.error(f"动态规划错误: {str(e)}\n{traceback.format_exc()}")
        
        # 回退机制：如果没有找到合适的组合，则使用最接近的单个钢材
        closest = min(self.module_steels, key=lambda m: abs(m['length'] - required_length))
        return [{'id': closest['id'], 'length': closest['length'], 'count': 1}]

    def _assign_modules(self, group_steels):
        """为设计钢材组分配模数钢材 - 使用改进的组合算法"""
        design_length = sum(s['length'] for s in group_steels)
        # 计算所需长度：设计长度 + (n-1)*焊接损耗
        required_length = design_length + self.params['weld_loss'] * (len(group_steels) - 1)
        
        # 尝试找到最匹配的模数钢材（单个）
        best_single = None
        best_single_diff = float('inf')
        for module in self.module_steels:
            diff = abs(module['length'] - required_length)
            if diff <= self.params['tolerance'] and diff < best_single_diff:
                best_single = [{'id': module['id'], 'length': module['length'], 'count': 1}]
                best_single_diff = diff
        
        if best_single:
            return best_single
        
        # 否则使用动态规划找到最佳组合
        return self._find_best_combination(required_length)

    def _calculate_loss(self, solution):
        """计算解决方案的损耗率"""
        total_design_length = 0
        total_module_length = 0
        
        for group in solution:
            total_design_length += group['design_length']
            for m in group['module_steels']:
                total_module_length += m['length'] * m['count']
        
        if total_module_length == 0:
            return float('inf')
        
        return (total_module_length - total_design_length) / total_module_length * 100

    def optimize(self):
        """执行优化算法 - 改进遗传算法"""
        global stop_optimization
        
        # 记录优化参数
        logging.info(f"开始优化，参数: {self.params}")
        logging.info(f"设计钢材数量: {len(self.design_steels)}")
        logging.info(f"模数钢材: {self.module_steels}")
        
        population_size = 30
        population = [self._generate_random_solution() for _ in range(population_size)]
        max_generations_without_improvement = 15
        
        max_time = self.params.get('max_time', 300)
        target_loss = self.params.get('target_loss', 10)
        
        logging.info(f"开始优化: 最大时间 {max_time}秒, 目标损耗率 {target_loss}%")
        
        start_time = time.time()
        
        while time.time() - start_time < max_time and self.no_improvement_count < max_generations_without_improvement:
            if stop_optimization:
                self.stop_reason = "用户手动停止"
                logging.info("优化被用户停止")
                break
                
            self.generation += 1
            
            # 评估种群
            evaluated = []
            for solution in population:
                loss_rate = self._calculate_loss(solution)
                evaluated.append((solution, loss_rate))
            
            # 排序 (损耗率越低越好)
            evaluated.sort(key=lambda x: x[1])
            
            # 更新最佳方案
            current_best_solution, current_best_loss = evaluated[0]
            if current_best_loss < self.best_loss:
                self.best_solution = current_best_solution
                self.best_loss = current_best_loss
                self.no_improvement_count = 0
                logging.info(f"第{self.generation}代: 发现更好方案 {current_best_loss:.2f}%")
                
                # 检查是否达到期望损耗率
                if current_best_loss <= target_loss:
                    self.stop_reason = f"达到目标损耗率 {target_loss}%"
                    logging.info(self.stop_reason)
                    break
            else:
                self.no_improvement_count += 1
            
            # 选择前50%作为精英
            elite_size = max(4, int(len(population) * 0.4))
            elite = [sol for sol, _ in evaluated[:elite_size]]
            
            # 创建新一代
            new_population = elite.copy()
            while len(new_population) < len(population):
                # 选择父代：80%概率从精英中选择，20%概率从普通中选择
                if random.random() < 0.8:
                    parent1, parent2 = random.choices(elite, k=2)
                else:
                    parent1, parent2 = random.choices(population, k=2)
                
                child = self._crossover(parent1, parent2)
                child = self._mutate(child)
                new_population.append(child)
            
            population = new_population
        
        # 计算最终结果
        calc_time = time.time() - start_time
        
        if self.stop_reason == "优化完成":
            if self.no_improvement_count >= max_generations_without_improvement:
                self.stop_reason = f"连续{max_generations_without_improvement}代无改进"
            else:
                self.stop_reason = f"达到最大计算时间 {max_time}秒"
        
        logging.info(f"优化完成: {self.stop_reason}, 耗时 {calc_time:.2f}秒, 最佳损耗率 {self.best_loss:.2f}%")
        
        result = {
            'loss_rate': self.best_loss,
            'cost_saving': self._calculate_cost_saving(self.best_solution),
            'calc_time': calc_time,
            'combinations': [],
            'stop_reason': self.stop_reason
        }
        
        # 格式化组合详情
        for i, group in enumerate(self.best_solution):
            module_length = sum(m['length'] * m['count'] for m in group['module_steels'])
            difference = module_length - group['design_length']
            loss_rate = difference / module_length * 100 if module_length > 0 else 0
            
            result['combinations'].append({
                'group_id': f"G{i+1}",
                'design_steels': group['design_steels'],
                'design_length': group['design_length'],
                'module_steels': group['module_steels'],
                'module_length': module_length,
                'difference': difference,
                'loss_rate': loss_rate
            })
        
        return result

    def _calculate_cost_saving(self, solution):
        """计算节省成本 - 考虑材料密度和单价"""
        total_design_length = sum(group['design_length'] for group in solution)
        total_module_length = sum(
            sum(m['length'] * m['count'] for m in group['module_steels']) 
            for group in solution
        )
        
        # 获取材料密度和单价（从参数中获取）
        density = self.params.get('density', 7850)  # kg/m³, 默认钢材密度
        price_per_kg = self.params.get('price_per_kg', 5.0)  # 元/公斤, 默认价格
        
        # 计算重量差（转换为公斤）
        weight_difference_kg = (total_module_length - total_design_length) * density / 1000000
        
        # 计算成本节省
        return abs(weight_difference_kg) * price_per_kg

    def _crossover(self, parent1, parent2):
        """交叉操作 - 改进实现"""
        if not parent1 or not parent2 or len(parent1) < 2 or len(parent2) < 2:
            return parent1 or parent2
        
        # 随机选择切割点
        crossover_point1 = random.randint(1, len(parent1) - 1)
        crossover_point2 = random.randint(1, len(parent2) - 1)
        
        # 创建子代
        child = parent1[:crossover_point1] + parent2[crossover_point2:]
        
        # 确保所有钢材都被包含（避免丢失钢材）
        all_steels = set()
        for group in child:
            all_steels.update(group['design_steels'])
        
        # 检查是否有钢材丢失
        original_steels = {s['id'] for s in self.design_steels}
        missing_steels = original_steels - all_steels
        
        # 如果有钢材丢失，添加到随机组中
        if missing_steels:
            # 创建包含丢失钢材的新组
            for steel_id in missing_steels:
                # 找到钢材长度
                steel_length = self.design_length_map.get(steel_id, 0)
                new_group = {
                    'design_steels': [steel_id],
                    'design_length': steel_length,
                    'module_steels': self._assign_modules([{'id': steel_id, 'length': steel_length}])
                }
                # 随机插入位置
                insert_pos = random.randint(0, len(child))
                child.insert(insert_pos, new_group)
        
        return child

    def _mutate(self, solution):
        """变异操作 - 改进实现"""
        if random.random() < 0.4 and len(solution) >= 2:  # 40%变异概率
            # 随机选择变异类型
            mutation_type = random.choice(['swap', 'split', 'merge'])
            
            if mutation_type == 'swap':
                # 随机交换两个组中的钢材
                idx1, idx2 = random.sample(range(len(solution)), 2)
                group1 = solution[idx1]
                group2 = solution[idx2]
                
                if group1['design_steels'] and group2['design_steels']:
                    # 随机选择钢材交换
                    steel_idx1 = random.randint(0, len(group1['design_steels']) - 1)
                    steel_idx2 = random.randint(0, len(group2['design_steels']) - 1)
                    
                    # 交换钢材
                    steel1 = group1['design_steels'][steel_idx1]
                    steel2 = group2['design_steels'][steel_idx2]
                    
                    group1['design_steels'][steel_idx1] = steel2
                    group2['design_steels'][steel_idx2] = steel1
                    
                    # 重新计算组属性
                    group1['design_length'] = sum(self.design_length_map[sid] for sid in group1['design_steels'])
                    group2['design_length'] = sum(self.design_length_map[sid] for sid in group2['design_steels'])
                    
                    # 重新分配模数钢材
                    group1['module_steels'] = self._assign_modules([
                        {'id': sid, 'length': self.design_length_map.get(sid, 0)} 
                        for sid in group1['design_steels']
                    ])
                    group2['module_steels'] = self._assign_modules([
                        {'id': sid, 'length': self.design_length_map.get(sid, 0)} 
                        for sid in group2['design_steels']
                    ])
            
            elif mutation_type == 'split' and len(solution) < 20:  # 限制最大组数
                # 随机选择一个组进行拆分
                idx = random.randint(0, len(solution) - 1)
                group = solution[idx]
                
                if len(group['design_steels']) > 1:
                    # 随机选择拆分点
                    split_point = random.randint(1, len(group['design_steels']) - 1)
                    group1_steels = group['design_steels'][:split_point]
                    group2_steels = group['design_steels'][split_point:]
                    
                    # 创建新组
                    new_group1 = {
                        'design_steels': group1_steels,
                        'design_length': sum(self.design_length_map[sid] for sid in group1_steels),
                        'module_steels': self._assign_modules([
                            {'id': sid, 'length': self.design_length_map.get(sid, 0)} 
                            for sid in group1_steels
                        ])
                    }
                    new_group2 = {
                        'design_steels': group2_steels,
                        'design_length': sum(self.design_length_map[sid] for sid in group2_steels),
                        'module_steels': self._assign_modules([
                            {'id': sid, 'length': self.design_length_map.get(sid, 0)} 
                            for sid in group2_steels
                        ])
                    }
                    
                    # 替换原组
                    solution[idx] = new_group1
                    solution.insert(idx + 1, new_group2)
            
            elif mutation_type == 'merge' and len(solution) > 1:
                # 随机选择两个相邻组进行合并
                idx = random.randint(0, len(solution) - 2)
                group1 = solution[idx]
                group2 = solution[idx + 1]
                
                # 合并钢材
                merged_steels = group1['design_steels'] + group2['design_steels']
                merged_group = {
                    'design_steels': merged_steels,
                    'design_length': group1['design_length'] + group2['design_length'],
                    'module_steels': self._assign_modules([
                        {'id': sid, 'length': self.design_length_map.get(sid, 0)} 
                        for sid in merged_steels
                    ])
                }
                
                # 替换原组
                solution[idx] = merged_group
                del solution[idx + 1]
        
        return solution
