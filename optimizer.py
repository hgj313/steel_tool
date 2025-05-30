import time
import random
import sqlite3
from datetime import datetime

class SteelOptimizer:
    def __init__(self, design_steels, module_steels, params):
        self.design_steels = self._expand_design_steels(design_steels)
        self.module_steels = module_steels
        self.params = params
        self.best_solution = None
        self.R_solution = None
        self.start_time = time.time()
        self.progress = 0
        self.status = "准备中"
    
    def _expand_design_steels(self, design_steels):
        """将设计钢材展开为单体列表"""
        expanded = []
        counter = 1
        for steel in design_steels:
            for i in range(steel['quantity']):
                expanded.append({
                    'id': f"a{counter}",
                    'length': steel['length'],
                    'original_id': steel['id']
                })
                counter += 1
        return expanded
    
    def _generate_random_solution(self):
        """生成随机初始解决方案"""
        solution = []
        remaining_steels = self.design_steels.copy()
        random.shuffle(remaining_steels)
        
        # 创建随机分组
        while remaining_steels:
            group_size = random.randint(1, min(5, len(remaining_steels)))
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
    
    def _assign_modules(self, group_steels):
        """为设计钢材组分配模数钢材"""
        design_length = sum(s['length'] for s in group_steels)
        required_length = design_length + self.params['cut_loss'] * (len(group_steels) - 1)
        
        # 尝试找到最匹配的模数钢材
        best_fit = None
        best_diff = float('inf')
        
        for module in self.module_steels:
            diff = abs(module['length'] - required_length)
            if diff <= self.params['tolerance'] and diff < best_diff:
                best_fit = [{'id': module['id'], 'length': module['length'], 'count': 1}]
                best_diff = diff
        
        # 如果找到匹配的单个模数钢材
        if best_fit:
            return best_fit
        
        # 否则尝试组合模数钢材
        # 简化实现 - 实际应使用更优算法
        combinations = self._generate_module_combinations(required_length)
        if combinations:
            return min(combinations, key=lambda c: abs(c['total_length'] - required_length))['modules']
        
        # 没有合适组合，使用最接近的单个模数钢材
        closest = min(self.module_steels, key=lambda m: abs(m['length'] - required_length))
        return [{'id': closest['id'], 'length': closest['length'], 'count': 1}]
    
    def _generate_module_combinations(self, required_length):
        """生成可能的模数钢材组合"""
        combinations = []
        modules_sorted = sorted(self.module_steels, key=lambda m: m['length'], reverse=True)
        
        # 简单贪心算法
        current_length = 0
        current_modules = []
        
        for module in modules_sorted:
            while current_length + module['length'] <= required_length + self.params['tolerance']:
                current_modules.append({'id': module['id'], 'length': module['length'], 'count': 1})
                current_length += module['length']
                
                if current_length >= required_length - self.params['tolerance']:
                    combinations.append({
                        'modules': current_modules.copy(),
                        'total_length': current_length
                    })
        
        return combinations
    
    def _calculate_loss(self, solution):
        """计算解决方案的损耗率"""
        total_design_length = 0
        total_module_length = 0
        
        for group in solution:
            total_design_length += group['design_length']
            total_module_length += sum(m['length'] * m['count'] for m in group['module_steels'])
        
        return (total_module_length - total_design_length) / total_module_length * 100
    
    def optimize(self):
        """执行优化算法"""
        self.status = "优化中"
        population = [self._generate_random_solution() for _ in range(20)]
        generation = 0
        
        while time.time() - self.start_time < self.params['max_time']:
            generation += 1
            
            # 评估种群
            evaluated = []
            for solution in population:
                loss_rate = self._calculate_loss(solution)
                evaluated.append((solution, loss_rate))
            
            # 排序 (损耗率越低越好)
            evaluated.sort(key=lambda x: x[1])
            
            # 更新最佳方案
            current_best = evaluated[0]
            if not self.best_solution or current_best[1] < self.best_solution[1]:
                self.best_solution = current_best
                
                # 检查是否达到期望损耗率
                if current_best[1] <= self.params['target_loss'] and not self.R_solution:
                    self.R_solution = current_best
            
            # 选择前50%作为精英
            elite_size = int(len(population) * 0.5)
            elite = [sol for sol, _ in evaluated[:elite_size]]
            
            # 创建新一代
            new_population = elite.copy()
            while len(new_population) < len(population):
                parent1, parent2 = random.choices(elite, k=2)
                child = self._crossover(parent1, parent2)
                child = self._mutate(child)
                new_population.append(child)
            
            population = new_population
            
            # 更新进度
            elapsed = time.time() - self.start_time
            self.progress = min(100, int(elapsed / self.params['max_time'] * 100))
        
        # 返回最佳解决方案
        self.status = "完成"
        calc_time = time.time() - self.start_time
        
        result = {
            'loss_rate': self.best_solution[1],
            'cost_saving': self._calculate_cost_saving(self.best_solution),
            'calc_time': calc_time,
            'combinations': []
        }
        
        # 格式化组合详情
        for i, group in enumerate(self.best_solution[0]):
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
        """计算节省成本（简化实现）"""
        total_design_length = sum(group['design_length'] for group in solution[0])
        total_module_length = sum(
            sum(m['length'] * m['count'] for m in group['module_steels']) 
            for group in solution[0]
        )
        
        # 假设每毫米成本（实际应根据参数计算）
        cost_per_mm = 0.05
        return (total_design_length - total_module_length) * cost_per_mm
    
    def _crossover(self, parent1, parent2):
        """交叉操作"""
        # 简化实现 - 随机选择父代的部分组
        crossover_point = random.randint(1, min(len(parent1), len(parent2)) - 1)
        child = parent1[:crossover_point] + parent2[crossover_point:]
        return child
    
    def _mutate(self, solution):
        """变异操作"""
        # 随机修改一个组
        if random.random() < 0.3:  # 30%变异概率
            idx = random.randint(0, len(solution) - 1)
            group = solution[idx]
            
            # 随机添加或移除一个设计钢材
            if random.random() < 0.5 and len(group['design_steels']) > 1:
                # 移除一个钢材
                remove_idx = random.randint(0, len(group['design_steels']) - 1)
                del group['design_steels'][remove_idx]
            else:
                # 添加一个钢材（从其他组随机取）
                other_groups = [g for g in solution if g != group]
                if other_groups:
                    donor = random.choice(other_groups)
                    if donor['design_steels']:
                        steel_idx = random.randint(0, len(donor['design_steels']) - 1)
                        steel = donor['design_steels'][steel_idx]
                        group['design_steels'].append(steel)
                        del donor['design_steels'][steel_idx]
            
            # 重新计算组属性
            group['design_length'] = self._calculate_group_length(group['design_steels'])
            group['module_steels'] = self._assign_modules(group['design_steels'])
        
        return solution
    
    def _calculate_group_length(self, design_steels):
        """计算设计钢材组的总长度"""
        # 从设计钢材列表中获取长度
        return sum(float(steel.split('-')[1]) for steel in design_steels)  # 简化实现
