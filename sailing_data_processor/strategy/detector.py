"""
セーリング戦略分析システム - 戦略ポイント検出アルゴリズム

風向シフト、タック、レイラインなどの戦略的判断ポイントを
検出するアルゴリズムを提供します。
"""

import numpy as np
import math
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
from functools import lru_cache

# 内部モジュールのインポート
from .points import StrategyPoint, WindShiftPoint, TackPoint, LaylinePoint, StrategyAlternative

try:
    from ..utilities.math_utils import normalize_angle, angle_difference
except ImportError:
    # スタンドアロン実行用
    def normalize_angle(angle):
        """角度を0-360度の範囲に正規化"""
        return angle % 360
        
    def angle_difference(angle1, angle2):
        """2つの角度間の最小差分を計算"""
        return ((angle1 - angle2 + 180) % 360) - 180


class StrategyDetector:
    """戦略ポイント検出クラス"""
    
    def __init__(self, wind_field_interpolator=None, vmg_calculator=None):
        """
        初期化
        
        Parameters:
        -----------
        wind_field_interpolator : WindFieldInterpolator
            風の場補間器（オプション）
        vmg_calculator : OptimalVMGCalculator
            VMG計算機（オプション）
        """
        self.wind_field_interpolator = wind_field_interpolator
        self.vmg_calculator = vmg_calculator
        
        # 検出設定
        self.config = {
            # 全般設定
            'min_confidence_threshold': 0.6,    # 信頼度の最小閾値
            'max_strategy_points': 8,           # 最大戦略ポイント数
            
            # 風向シフト検出設定
            'min_wind_shift_angle': 5.0,        # 最小風向シフト角度（度）
            'wind_forecast_interval': 300,      # 風予測間隔（秒）
            'max_wind_forecast_horizon': 1800,  # 最大風予測期間（秒）
            
            # タック検出設定
            'tack_search_radius': 500,          # タック探索半径（メートル）
            'min_vmg_improvement': 0.05,        # 最小VMG改善閾値（比率）
            'max_tacks_per_leg': 3,             # レグあたり最大タック数
            
            # レイライン検出設定
            'layline_safety_margin': 10.0,      # レイライン安全マージン（度）
            'min_mark_distance': 100,           # マークからの最小検出距離（メートル）
        }
    
    def detect_wind_shifts(self, course_data: Dict[str, Any], wind_field: Dict[str, Any], 
                         target_time: datetime = None) -> List[WindShiftPoint]:
        """
        風向シフトポイントを検出
        
        Parameters:
        -----------
        course_data : Dict[str, Any]
            コース計算結果
        wind_field : Dict[str, Any]
            風の場データ
        target_time : datetime, optional
            検出対象時間
            
        Returns:
        --------
        List[WindShiftPoint]
            検出された風向シフトポイント
        """
        wind_shift_points = []
        
        # 各レグを処理
        for leg_idx, leg in enumerate(course_data.get('legs', [])):
            path_points = leg.get('path', {}).get('path_points', [])
            
            if not path_points:
                continue
                
            # パスポイントをサンプリング（効率化のため）
            sampling_rate = max(1, len(path_points) // 20)  # 約20ポイントをサンプリング
            
            # 風向の履歴を追跡
            direction_history = []
            position_history = []
            time_history = []
            
            for i in range(0, len(path_points), sampling_rate):
                point = path_points[i]
                
                # 現在位置と予測到達時間
                lat, lon = point.get('lat'), point.get('lon')
                point_time = point.get('time', 0)
                
                # 現在の風情報を取得
                current_wind = self._get_wind_at_position(lat, lon, target_time, wind_field)
                if not current_wind:
                    continue
                
                # 風向履歴を追加
                direction_history.append(current_wind.get('direction', 0))
                position_history.append((lat, lon))
                time_history.append(point_time)
                
                # 風向トレンド分析（少なくとも3点必要）
                if len(direction_history) >= 3:
                    # 過去3点の風向変化を分析
                    dir_changes = [angle_difference(direction_history[j], direction_history[j-1]) 
                                 for j in range(1, len(direction_history))]
                    
                    # 一貫した風向変化を検出
                    if len(dir_changes) >= 2:
                        consistent_change = all(c > 0 for c in dir_changes) or all(c < 0 for c in dir_changes)
                        cumulative_change = sum(dir_changes)
                        
                        # 有意な風向変化を検出
                        min_shift = self.config['min_wind_shift_angle']
                        if abs(cumulative_change) >= min_shift and consistent_change:
                            # 風向シフトポイントを作成
                            shift_point = WindShiftPoint(position_history[-2], time_history[-2])
                            shift_point.shift_angle = cumulative_change
                            shift_point.shift_duration = time_history[-1] - time_history[0]
                            
                            # 信頼度計算
                            base_confidence = 0.5 + min(0.4, abs(cumulative_change) / 30)
                            variability_factor = 1.0 - current_wind.get('variability', 0.2)
                            shift_point.shift_probability = base_confidence * variability_factor
                            
                            # 有利な風向シフトかどうかを判定
                            current_course = point.get('course', 0)
                            is_upwind = leg.get('is_upwind', False)
                            shift_point.favorable = self._is_favorable_shift(
                                cumulative_change, current_course, is_upwind
                            )
                            
                            # 風情報を設定
                            shift_point.wind_info = {
                                'direction': current_wind.get('direction', 0),
                                'speed': current_wind.get('speed', 0),
                                'direction_trend': cumulative_change / max(1, shift_point.shift_duration / 60),
                                'confidence': shift_point.shift_probability,
                                'variability': current_wind.get('variability', 0.2)
                            }
                            
                            # 説明を生成
                            shift_type = "favorable" if shift_point.favorable else "unfavorable"
                            leg_name = leg.get('leg_number', leg_idx + 1)
                            shift_point.description = (
                                f"Leg {leg_name}: {abs(cumulative_change):.1f}° {shift_type} wind shift trend "
                                f"detected. Wind: {current_wind.get('direction'):.1f}°/{current_wind.get('speed'):.1f}kts."
                            )
                            
                            # 推奨アクション
                            if shift_point.favorable:
                                shift_point.recommendation = (
                                    f"Prepare to take advantage of this favorable shift. "
                                    f"Consider {'staying on current tack' if cumulative_change > 0 else 'tacking early'}."
                                )
                            else:
                                shift_point.recommendation = (
                                    f"Prepare to minimize impact of this unfavorable shift. "
                                    f"Consider {'tacking early' if cumulative_change > 0 else 'staying on current tack'}."
                                )
                            
                            wind_shift_points.append(shift_point)
                            
                            # 履歴をリセット（重複検出防止）
                            direction_history = direction_history[-1:]
                            position_history = position_history[-1:]
                            time_history = time_history[-1:]
                
                # シフト予測の実行
                # 短期、中期、長期の予測時間枠で風向変化を予測
                prediction_horizons = [300, 600, 1200]  # 秒単位（5分、10分、20分）
                
                for horizon in prediction_horizons:
                    # 将来の時点
                    future_time_point = None
                    if target_time:
                        future_time_point = target_time + timedelta(seconds=horizon)
                    else:
                        future_time_point = point_time + horizon
                    
                    # 将来の風情報を取得
                    future_wind = self._get_wind_at_position(lat, lon, future_time_point, wind_field)
                    if not future_wind:
                        continue
                    
                    # 風向差を計算
                    dir_diff = angle_difference(
                        current_wind.get('direction', 0),
                        future_wind.get('direction', 0)
                    )
                    
                    # 最小シフト角を超える変化を検出
                    if abs(dir_diff) >= self.config['min_wind_shift_angle']:
                        # 信頼度スケーリング（予測時間が長いと信頼度は下がる）
                        horizon_factor = max(0.5, 1.0 - horizon / 3600)  # 1時間で半分の信頼度に
                        
                        # 風向シフトポイントを作成
                        shift_point = WindShiftPoint((lat, lon), point_time)
                        shift_point.shift_angle = dir_diff
                        shift_point.shift_duration = horizon
                        shift_point.shift_probability = current_wind.get('confidence', 0.8) * \
                                                     future_wind.get('confidence', 0.7) * \
                                                     horizon_factor
                        
                        # 有利な風向シフトかどうかを判定
                        current_course = point.get('course', 0)
                        is_upwind = leg.get('is_upwind', False)
                        shift_point.favorable = self._is_favorable_shift(
                            dir_diff, current_course, is_upwind
                        )
                        
                        # 風情報を設定
                        shift_point.wind_info = {
                            'direction': current_wind.get('direction', 0),
                            'speed': current_wind.get('speed', 0),
                            'direction_trend': dir_diff / (horizon / 60),  # 度/分
                            'speed_trend': (future_wind.get('speed', 0) - current_wind.get('speed', 0)) / (horizon / 60),
                            'confidence': shift_point.shift_probability,
                            'variability': current_wind.get('variability', 0.2)
                        }
                        
                        # 説明を生成
                        shift_type = "favorable" if shift_point.favorable else "unfavorable"
                        leg_name = leg.get('leg_number', leg_idx + 1)
                        shift_point.description = (
                            f"Leg {leg_name}: {abs(dir_diff):.1f}° {shift_type} wind shift predicted "
                            f"in {horizon//60} minutes. Current wind: {current_wind.get('direction'):.1f}°/{current_wind.get('speed'):.1f}kts."
                        )
                        
                        # 推奨アクション
                        time_to_decision = max(1, horizon // 60 - 2)  # 2分前には判断が必要
                        
                        if shift_point.favorable:
                            if is_upwind:
                                # 風上向かいレグでの推奨
                                shift_point.recommendation = (
                                    f"Reassess in {time_to_decision} minutes. Consider "
                                    f"{'staying on current tack' if shift_point.shift_angle > 0 else 'tacking'} "
                                    f"to maximize advantage."
                                )
                            else:
                                # 風下レグでの推奨
                                shift_point.recommendation = (
                                    f"Reassess in {time_to_decision} minutes. Consider "
                                    f"{'gybing' if shift_point.shift_angle > 0 else 'staying on current course'} "
                                    f"to maximize advantage."
                                )
                        else:
                            if is_upwind:
                                # 風上向かいレグでの推奨
                                shift_point.recommendation = (
                                    f"Reassess in {time_to_decision} minutes. Consider "
                                    f"{'tacking' if shift_point.shift_angle > 0 else 'staying on current tack'} "
                                    f"to minimize impact."
                                )
                            else:
                                # 風下レグでの推奨
                                shift_point.recommendation = (
                                    f"Reassess in {time_to_decision} minutes. Consider "
                                    f"{'staying on current course' if shift_point.shift_angle > 0 else 'gybing'} "
                                    f"to minimize impact."
                                )
                        
                        wind_shift_points.append(shift_point)
                        break  # 最初の有意なシフトのみ対象
        
        # 重複するシフトを除去（位置と時間が近いもの）
        filtered_points = self._filter_duplicate_shifts(wind_shift_points)
        
        return filtered_points
    
    def detect_optimal_tacks(self, course_data: Dict[str, Any], wind_field: Dict[str, Any]) -> List[TackPoint]:
        """
        最適タックポイントを検出
        
        Parameters:
        -----------
        course_data : Dict[str, Any]
            コース計算結果
        wind_field : Dict[str, Any]
            風の場データ
            
        Returns:
        --------
        List[TackPoint]
            検出された最適タックポイント
        """
        tack_points = []
        
        # VMGCalculatorが必要
        if not self.vmg_calculator:
            warnings.warn("VMGCalculatorが設定されていないため、最適タックポイントの検出ができません")
            return tack_points
        
        # 各レグを処理
        for leg_idx, leg in enumerate(course_data.get('legs', [])):
            # レグの基本情報を取得
            leg_path = leg.get('path', {})
            path_points = leg_path.get('path_points', [])
            
            if not path_points:
                continue
            
            # レグの目標位置（終点ウェイポイント）
            target_waypoint = leg.get('end_waypoint', {})
            target_pos = (target_waypoint.get('lat', 0), target_waypoint.get('lon', 0))
            
            # レグタイプを確認 (風上/風下)
            is_upwind = leg.get('is_upwind', False)
            
            # パスポイントをサンプリング（効率化のため）
            sampling_rate = max(1, len(path_points) // 15)  # 約15ポイントをサンプリング
            
            # 既存のタックポイントを取得（参考情報として）
            existing_tacks = leg_path.get('tack_points', [])
            existing_tack_positions = [(t.get('lat', 0), t.get('lon', 0)) for t in existing_tacks]
            
            # 重要ポイントでタック評価
            for i in range(0, len(path_points), sampling_rate):
                point = path_points[i]
                
                # 現在位置と予測到達時間
                lat, lon = point.get('lat'), point.get('lon')
                current_pos = (lat, lon)
                current_time = point.get('time', 0)
                
                # 残りの距離が短い場合はタック評価を省略
                distance_to_target = self._calculate_distance(lat, lon, target_pos[0], target_pos[1])
                if distance_to_target < self.config['min_mark_distance']:
                    continue
                
                # 現在の風情報を取得
                current_wind = self._get_wind_at_position(lat, lon, current_time, wind_field)
                if not current_wind:
                    continue
                
                # 現在のコースとVMG情報
                current_vmg_info = self.vmg_calculator.calculate_optimal_vmg(
                    boat_type='default',  # 設定から取得するべき
                    lat=lat, 
                    lon=lon,
                    target_lat=target_pos[0], 
                    target_lon=target_pos[1]
                )
                
                if not current_vmg_info:
                    continue
                
                # 現在のコースと速度
                current_course = current_vmg_info.get('optimal_course', 0)
                current_speed = current_vmg_info.get('boat_speed', 0)
                
                # タックが必要かどうか評価
                tack_needed = current_vmg_info.get('tack_needed', False)
                
                # タックが必要な場合、最適なタック位置を探索
                if tack_needed:
                    # 最適タック位置を探索
                    optimal_tack = self._find_optimal_tack_position(
                        current_pos=current_pos,
                        target_pos=target_pos,
                        current_time=current_time,
                        current_vmg_info=current_vmg_info,
                        wind_field=wind_field,
                        search_radius=self.config['tack_search_radius']
                    )
                    
                    if optimal_tack:
                        # 既存タック位置との重複をチェック
                        is_duplicate = False
                        for existing_pos in existing_tack_positions:
                            dist = self._calculate_distance(
                                optimal_tack['position'][0], optimal_tack['position'][1],
                                existing_pos[0], existing_pos[1]
                            )
                            if dist < 200:  # 200m以内なら重複とみなす
                                is_duplicate = True
                                break
                        
                        if not is_duplicate:
                            # TackPointオブジェクトを作成
                            tack_point = TackPoint(
                                position=optimal_tack['position'],
                                time_estimate=optimal_tack['time']
                            )
                            
                            # タック情報を設定
                            tack_point.tack_angle = abs(angle_difference(
                                current_course, optimal_tack['after_course']
                            ))
                            tack_point.vmg_gain = optimal_tack['vmg_gain']
                            tack_point.timing_sensitivity = self._calculate_tack_timing_sensitivity(
                                optimal_tack['position'],
                                current_wind,
                                is_upwind
                            )
                            
                            # 風情報を設定
                            tack_point.wind_info = {
                                'direction': current_wind.get('direction', 0),
                                'speed': current_wind.get('speed', 0),
                                'confidence': current_wind.get('confidence', 0.8),
                                'variability': current_wind.get('variability', 0.2)
                            }
                            
                            # 説明と推奨アクションを生成
                            tack_type = "Tack" if is_upwind else "Gybe"
                            leg_name = leg.get('leg_number', leg_idx + 1)
                            
                            tack_point.description = (
                                f"Leg {leg_name}: Optimal {tack_type.lower()} point with "
                                f"{tack_point.vmg_gain*100:.1f}% VMG improvement. "
                                f"Wind: {current_wind.get('direction'):.1f}°/{current_wind.get('speed'):.1f}kts."
                            )
                            
                            if tack_point.timing_sensitivity > 0.7:
                                tack_point.recommendation = (
                                    f"Execute {tack_type.lower()} with precise timing. "
                                    f"Expected new course: {optimal_tack['after_course']:.1f}°."
                                )
                            else:
                                tack_point.recommendation = (
                                    f"Execute {tack_type.lower()} when ready. "
                                    f"Expected new course: {optimal_tack['after_course']:.1f}°."
                                )
                            
                            # リスク評価
                            tack_point.evaluate_risk()
                            
                            tack_points.append(tack_point)
        
        # 重複したタックポイントを除去（近接したもの）
        filtered_tacks = self._filter_duplicate_tacks(tack_points)
        
        return filtered_tacks
    
    def detect_laylines(self, course_data: Dict[str, Any], wind_field: Dict[str, Any]) -> List[LaylinePoint]:
        """
        レイラインポイントを検出
        
        Parameters:
        -----------
        course_data : Dict[str, Any]
            コース計算結果
        wind_field : Dict[str, Any]
            風の場データ
            
        Returns:
        --------
        List[LaylinePoint]
            検出されたレイラインポイント
        """
        layline_points = []
        
        # VMGCalculatorが必要
        if not self.vmg_calculator:
            warnings.warn("VMGCalculatorが設定されていないため、レイラインポイントの検出ができません")
            return layline_points
        
        # 各レグを処理
        for leg_idx, leg in enumerate(course_data.get('legs', [])):
            # 風上レグのみ処理（風下ではレイラインはあまり重要でない）
            if not leg.get('is_upwind', False):
                continue
            
            # マーク情報を取得
            end_waypoint = leg.get('end_waypoint')
            if not end_waypoint:
                continue
                
            mark_pos = (end_waypoint.get('lat'), end_waypoint.get('lon'))
            mark_name = end_waypoint.get('name', f"Mark {leg_idx+1}")
            
            # パスポイントを取得
            path_points = leg.get('path', {}).get('path_points', [])
            if not path_points:
                continue
            
            # マークでの風情報を取得
            mark_wind = self._get_wind_at_position(mark_pos[0], mark_pos[1], None, wind_field)
            if not mark_wind:
                continue
                
            # 最小マーク距離（近すぎるとレイラインが重要でなくなる）
            min_distance = self.config['min_mark_distance']
            
            # 各パスポイントでレイライン評価
            for i, point in enumerate(path_points):
                # 位置と時間を取得
                lat, lon = point.get('lat'), point.get('lon')
                point_time = point.get('time', 0)
                current_pos = (lat, lon)
                
                # マークまでの距離を計算
                dist_to_mark = self._calculate_distance(lat, lon, mark_pos[0], mark_pos[1])
                
                # 最小距離チェック
                if dist_to_mark < min_distance:
                    continue
                    
                # 現在位置の風情報
                point_wind = self._get_wind_at_position(lat, lon, point_time, wind_field)
                if not point_wind:
                    continue
                    
                # レイライン計算
                layline_info = self._calculate_layline(
                    current_pos, mark_pos, point_time, point_wind, wind_field
                )
                
                if not layline_info:
                    continue
                    
                # レイライン到達判定
                if layline_info['on_port_layline'] or layline_info['on_starboard_layline']:
                    # レイラインポイントを作成
                    layline_point = LaylinePoint(current_pos, point_time)
                    layline_point.mark_distance = dist_to_mark
                    
                    # レイラインタイプ（ポートかスターボード）
                    layline_type = "Port" if layline_info['on_port_layline'] else "Starboard"
                    layline_point.layline_angle = layline_info['port_layline'] if layline_type == "Port" else layline_info['starboard_layline']
                    
                    # 交通量係数の推定
                    layline_point.traffic_factor = self._estimate_traffic_factor(current_pos, mark_pos)
                    
                    # 風情報を設定
                    layline_point.wind_info = {
                        'direction': point_wind.get('direction', 0),
                        'speed': point_wind.get('speed', 0),
                        'confidence': point_wind.get('confidence', 0.8),
                        'variability': point_wind.get('variability', 0.2),
                        'direction_trend': layline_info['wind_shift'] / max(1, layline_info['time_to_arrival'] / 60)  # 度/分
                    }
                    
                    # 説明と推奨アクションを生成
                    layline_point.description = (
                        f"Leg {leg_idx+1}: {layline_type} layline to {mark_name}. "
                        f"Distance: {dist_to_mark:.0f}m. "
                        f"Wind: {point_wind.get('direction'):.1f}°/{point_wind.get('speed'):.1f}kts."
                    )
                    
                    # 安全マージンの情報を含める
                    layline_point.description += f" Safety margin: {layline_info['safety_margin']:.1f}°."
                    
                    # 風向シフト予測情報を含める
                    if abs(layline_info['wind_shift']) > 5:
                        shift_direction = "right" if layline_info['wind_shift'] > 0 else "left"
                        layline_point.description += f" Predicted {abs(layline_info['wind_shift']):.1f}° {shift_direction} shift before mark."
                    
                    # 推奨アクション
                    if dist_to_mark > 500:
                        layline_point.recommendation = (
                            f"Approaching {layline_type.lower()} layline. "
                            f"Consider {'extending outward' if layline_type == 'Port' else 'tacking to layline'} "
                            f"based on fleet position."
                        )
                    else:
                        # マークが近い場合はより明確な指示
                        layline_point.recommendation = (
                            f"On {layline_type.lower()} layline to mark. "
                            f"{'Prepare to tack' if layline_type == 'Starboard' else 'Continue on current tack'} "
                            f"to round the mark."
                        )
                    
                    # 信頼度計算
                    basic_confidence = point_wind.get('confidence', 0.8)
                    distance_factor = max(0.7, 1.0 - dist_to_mark / 2000)  # 2000m以上で0.7
                    shift_factor = max(0.6, 1.0 - abs(layline_info['wind_shift']) / 30)  # 30度以上のシフト予測で0.6
                    layline_point.confidence = basic_confidence * distance_factor * shift_factor
                    
                    # リスク評価
                    layline_point.evaluate_risk()
                    
                    layline_points.append(layline_point)
        
        # 重複するレイラインポイントを除去
        filtered_points = self._filter_duplicate_laylines(layline_points)
        
        return filtered_points

    def detect_critical_points_optimized(self, course_data: Dict[str, Any], wind_field: Dict[str, Any], 
                                      target_time: datetime = None) -> List[StrategyPoint]:
        """
        大規模コースデータの効率的な戦略ポイント検出
        
        Parameters:
        -----------
        course_data : Dict[str, Any]
            コース計算結果
        wind_field : Dict[str, Any]
            風の場データ
        target_time : datetime, optional
            戦略ポイント検出の対象時間
            
        Returns:
        --------
        List[StrategyPoint]
            検出された戦略的判断ポイントのリスト
        """
        # フェーズ1: 重要レグの特定
        important_legs = self._identify_important_legs(course_data)
        
        # 全ての戦略ポイントを格納
        all_strategy_points = []
        
        # フェーズ2: 重要レグの詳細評価
        for leg_idx in important_legs:
            if leg_idx >= len(course_data.get('legs', [])):
                continue
                
            leg = course_data['legs'][leg_idx]
            
            # 詳細レベル（高）で戦略ポイント検出
            wind_shifts = self._detect_wind_shifts_for_leg(leg, wind_field, target_time, 'high')
            all_strategy_points.extend(wind_shifts)
            
            tack_points = self._detect_tacks_for_leg(leg, wind_field, 'high')
            all_strategy_points.extend(tack_points)
            
            layline_points = self._detect_laylines_for_leg(leg, wind_field, 'high')
            all_strategy_points.extend(layline_points)
        
        # フェーズ3: その他のレグの粗い評価
        other_legs = [i for i in range(len(course_data.get('legs', []))) if i not in important_legs]
        for leg_idx in other_legs:
            leg = course_data['legs'][leg_idx]
            
            # 詳細レベル（低）で戦略ポイント検出
            wind_shifts = self._detect_wind_shifts_for_leg(leg, wind_field, target_time, 'low')
            all_strategy_points.extend(wind_shifts)
            
            tack_points = self._detect_tacks_for_leg(leg, wind_field, 'low')
            all_strategy_points.extend(tack_points)
            
            layline_points = self._detect_laylines_for_leg(leg, wind_field, 'low')
            all_strategy_points.extend(layline_points)
        
        return all_strategy_points
    
    # ----- 内部ヘルパーメソッド -----
    
    def _find_optimal_tack_position(self, current_pos, target_pos, current_time, 
                                  current_vmg_info, wind_field, search_radius=500):
        """
        最適なタック位置を探索
        
        Parameters:
        -----------
        current_pos : Tuple[float, float]
            現在位置 (lat, lon)
        target_pos : Tuple[float, float]
            目標位置 (lat, lon)
        current_time : float
            現在時刻
        current_vmg_info : Dict[str, Any]
            現在のVMG情報
        wind_field : Dict[str, Any]
            風の場データ
        search_radius : float
            探索半径（メートル）
            
        Returns:
        --------
        Dict[str, Any] or None
            最適タック位置情報
        """
        # 探索ポイント生成
        search_points = []
        current_course = current_vmg_info.get('optimal_course', 0)
        
        # 現在コースに沿って複数ポイントを探索
        for distance in np.linspace(0, search_radius, 10):  # 10ポイントを探索
            search_lat, search_lon = self._get_point_at_distance(
                current_pos[0], current_pos[1], current_course, distance
            )
            search_points.append((search_lat, search_lon))
        
        # 評価結果を格納
        evaluation_results = []
        
        # 各ポイントでタック後のVMGを評価
        for point in search_points:
            # 予想到達時間を計算
            distance = self._calculate_distance(
                current_pos[0], current_pos[1], point[0], point[1]
            )
            boat_speed_ms = current_vmg_info.get('boat_speed', 5) * 0.51444  # ノットからm/sに変換
            time_delta = distance / boat_speed_ms if boat_speed_ms > 0 else 0
            point_time = current_time + time_delta
            
            # 到達時の風向風速を取得
            wind_at_point = self._get_wind_at_position(point[0], point[1], point_time, wind_field)
            if not wind_at_point:
                continue
            
            # タック後のVMG計算
            after_tack_vmg = self.vmg_calculator.calculate_optimal_vmg(
                boat_type='default',
                lat=point[0], 
                lon=point[1],
                target_lat=target_pos[0], 
                target_lon=target_pos[1]
            )
            
            if not after_tack_vmg:
                continue
            
            # タック操作によるVMGロスを考慮（タック直後は速度低下）
            tack_efficiency = 0.8  # タック直後は80%の効率と仮定
            adjusted_vmg = after_tack_vmg.get('vmg', 0) * tack_efficiency
            
            # 現在VMGからの改善率
            vmg_gain = (adjusted_vmg / current_vmg_info.get('vmg', 1)) - 1.0
            
            # VMG改善が最小閾値を超えるかチェック
            if vmg_gain < self.config['min_vmg_improvement']:
                continue
            
            # 残り距離の計算
            remaining_distance = self._calculate_distance(
                point[0], point[1], target_pos[0], target_pos[1]
            )
            
            # 総合スコア計算（VMG、残り距離、風の信頼度を考慮）
            score = (
                vmg_gain * 0.5 +                           # VMG利得の重み
                (1.0 / (remaining_distance + 1)) * 5000 * 0.3 +  # 残り距離の重み
                wind_at_point.get('confidence', 0.8) * 0.2      # 風の信頼度の重み
            )
            
            # 評価結果を追加
            evaluation_results.append({
                'position': point,
                'time': point_time,
                'vmg_after': adjusted_vmg,
                'after_course': after_tack_vmg.get('optimal_course', 0),
                'vmg_gain': vmg_gain,
                'confidence': wind_at_point.get('confidence', 0.8),
                'score': score
            })
        
        # 最適なポイントを選択
        if evaluation_results:
            best_result = max(evaluation_results, key=lambda x: x['score'])
            return best_result
        
        return None
    
    def _calculate_layline(self, position, mark_pos, time_point, current_wind, wind_field):
        """
        レイラインを計算
        
        Parameters:
        -----------
        position : Tuple[float, float]
            現在位置
        mark_pos : Tuple[float, float]
            マーク位置
        time_point : float
            現在時刻
        current_wind : Dict
            現在の風情報
        wind_field : Dict
            風の場データ
            
        Returns:
        --------
        Dict or None
            レイライン情報
        """
        if not self.vmg_calculator:
            return None
        
        # VMG情報を取得
        vmg_info = self.vmg_calculator.calculate_optimal_vmg(
            boat_type='default',
            lat=position[0], 
            lon=position[1],
            target_lat=mark_pos[0], 
            target_lon=mark_pos[1]
        )
        
        if not vmg_info:
            return None
        
        # マークまでの到達予想時間
        eta_seconds = vmg_info.get('eta_seconds', 0)
        
        # マーク到達時の風向予測
        arrival_time = time_point + eta_seconds
        arrival_wind = self._get_wind_at_position(
            mark_pos[0], mark_pos[1], arrival_time, wind_field
        )
        
        if not arrival_wind:
            arrival_wind = current_wind  # 予測できない場合は現在の風を使用
        
        # 風向変化を計算
        wind_shift = angle_difference(
            arrival_wind.get('direction', 0), 
            current_wind.get('direction', 0)
        )
        
        # 安全マージンを計算
        safety_margin = self._calculate_layline_safety_margin(
            position, current_wind, mark_pos, wind_shift
        )
        
        # 風向に基づくタッキング角度の計算
        boat_type = 'default'  # 設定から取得するべき
        
        # 最適タック角度を取得（風上角度を2倍してタック角度を概算）
        tacking_angle = self._get_optimal_tacking_angle(
            current_wind.get('speed', 10), 
            boat_type
        )
        
        # マークへの方位を計算
        bearing_to_mark = self._calculate_bearing(
            position[0], position[1], mark_pos[0], mark_pos[1]
        )
        
        # 現在の風向
        wind_direction = current_wind.get('direction', 0)
        
        # 両側のレイライン
        port_layline = normalize_angle(wind_direction - tacking_angle/2 + safety_margin)
        starboard_layline = normalize_angle(wind_direction + tacking_angle/2 - safety_margin)
        
        # コースとレイラインの角度差
        port_angle_diff = abs(angle_difference(bearing_to_mark, port_layline))
        starboard_angle_diff = abs(angle_difference(bearing_to_mark, starboard_layline))
        
        # レイライン到達判定
        on_port_layline = port_angle_diff < safety_margin/2
        on_starboard_layline = starboard_angle_diff < safety_margin/2
        
        return {
            'port_layline': port_layline,
            'starboard_layline': starboard_layline,
            'on_port_layline': on_port_layline,
            'on_starboard_layline': on_starboard_layline,
            'tacking_angle': tacking_angle,
            'safety_margin': safety_margin,
            'wind_shift': wind_shift,
            'time_to_arrival': eta_seconds,
            'current_wind': current_wind,
            'predicted_arrival_wind': arrival_wind
        }

    def _calculate_layline_safety_margin(self, position, wind_info, mark_pos, wind_shift=0):
        """
        レイラインの安全マージンを動的に計算
        
        Parameters:
        -----------
        position : Tuple[float, float]
            現在位置
        wind_info : Dict
            風情報
        mark_pos : Tuple[float, float]
            マーク位置
        wind_shift : float
            予測される風向変化
            
        Returns:
        --------
        float
            安全マージン（度）
        """
        # 基本マージン（艇種による）
        boat_type = 'default'  # 設定から取得するべき
        base_margin = {
            'default': 5.0,
            'laser': 6.0,
            '470': 5.0,
            '49er': 4.0,
            'finn': 6.5
        }.get(boat_type, 5.0)
        
        # 風の変動性に基づく調整
        variability = wind_info.get('variability', 0.2)
        variability_factor = 1.0 + (variability * 15.0)  # 変動が大きいほど大きなマージン
        
        # 風速に基づく調整（風が弱いほどマージン大きく）
        wind_speed = wind_info.get('speed', 10)
        speed_factor = 1.0 + max(0, (15 - wind_speed) / 10.0)
        
        # 信頼度に基づく調整（信頼度が低いほどマージン大きく）
        confidence = wind_info.get('confidence', 0.8)
        confidence_factor = 1.0 + ((1.0 - confidence) * 2.0)
        
        # マーク距離に基づく調整（近いほどマージン小さく）
        distance_to_mark = self._calculate_distance(
            position[0], position[1], mark_pos[0], mark_pos[1]
        )
        distance_factor = min(1.5, max(0.8, distance_to_mark / 500))  # 500mを基準に調整
        
        # 風向変化予測に基づく調整
        shift_factor = 1.0 + (abs(wind_shift) / 45.0)  # 45度の変化でマージン2倍
        
        # トラフィック要素（実装方法に応じて調整）
        traffic_factor = 1.0 + self._estimate_traffic_factor(position, mark_pos) * 0.5
        
        # 総合マージン計算
        margin = (
            base_margin * 
            variability_factor * 
            speed_factor * 
            confidence_factor * 
            distance_factor * 
            shift_factor *
            traffic_factor
        )
        
        # 適切な範囲に制限
        margin = min(25.0, max(3.0, margin))
        
        return margin
    
    def _estimate_traffic_factor(self, position, mark_pos, radius=100):
        """
        周辺の艇の密度を推定（トラフィック係数）
        
        Parameters:
        -----------
        position : Tuple[float, float]
            現在位置
        mark_pos : Tuple[float, float]
            マーク位置
        radius : float
            検知半径（メートル）
            
        Returns:
        --------
        float
            トラフィック係数（0-1）
        """
        # 実際の実装では、競合艇の位置データを使用
        # 簡易実装：マークに近いほどトラフィックが多いと仮定
        
        # マークに近いほどトラフィックが多いと仮定
        mark_distance = self._calculate_distance(
            position[0], position[1], mark_pos[0], mark_pos[1]
        )
        
        # 距離に基づく基本係数（マークに近いほど混雑）
        base_factor = max(0.1, min(0.9, 500 / max(1, mark_distance)))
        
        return base_factor
    
    def _get_optimal_tacking_angle(self, wind_speed, boat_type='default'):
        """
        最適なタック角度を計算（風速と艇種に基づく）
        
        Parameters:
        -----------
        wind_speed : float
            風速（ノット）
        boat_type : str
            艇種
            
        Returns:
        --------
        float
            タック角度（度）
        """
        # 各艇種の基本タック角度（風上の最適角度の2倍）
        base_angles = {
            'default': 90.0,
            'laser': 84.0,
            '470': 90.0,
            '49er': 85.0,
            'finn': 88.0
        }
        
        base_angle = base_angles.get(boat_type, 90.0)
        
        # 風速による調整（風が強いほどタイトな角度が可能）
        if wind_speed < 5:
            # 軽風時はタック角が広がる
            return base_angle + 10.0
        elif wind_speed > 15:
            # 強風時はタック角が狭まる
            return base_angle - 5.0
        else:
            # 中風域では線形に変化
            return base_angle + 10.0 - (wind_speed - 5) * 1.5
    
    def _filter_duplicate_shifts(self, wind_shift_points):
        """重複する風向シフトポイントを除去"""
        if len(wind_shift_points) <= 1:
            return wind_shift_points
        
        # 時間でソート
        sorted_points = sorted(wind_shift_points, key=lambda p: p.time_estimate)
        
        # フィルタリング済みのリスト
        filtered = []
        
        for i, point in enumerate(sorted_points):
            # 既存のポイントとの類似性をチェック
            is_duplicate = False
            
            for existing in filtered:
                # 位置の近さをチェック
                pos_distance = self._calculate_distance(
                    point.position[0], point.position[1],
                    existing.position[0], existing.position[1]
                )
                
                # 時間の近さをチェック
                time_diff = abs(point.time_estimate - existing.time_estimate)
                
                # 位置と時間が両方近い場合は重複とみなす
                if pos_distance < 500 and time_diff < 300:  # 500m以内、5分以内
                    # 信頼度の高い方を保持
                    if point.shift_probability > existing.shift_probability:
                        # 既存のポイントを置き換え
                        filtered.remove(existing)
                        filtered.append(point)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered.append(point)
        
        return filtered
    
    def _filter_duplicate_tacks(self, tack_points):
        """重複するタックポイントを除去"""
        if len(tack_points) <= 1:
            return tack_points
        
        # 時間でソート
        sorted_points = sorted(tack_points, key=lambda p: p.time_estimate)
        
        # フィルタリング済みのリスト
        filtered = []
        
        for point in sorted_points:
            # 既存のポイントとの類似性をチェック
            is_duplicate = False
            
            for existing in filtered:
                # 位置の近さをチェック
                pos_distance = self._calculate_distance(
                    point.position[0], point.position[1],
                    existing.position[0], existing.position[1]
                )
                
                # 時間の近さをチェック
                time_diff = abs(point.time_estimate - existing.time_estimate)
                
                # 位置と時間が両方近い場合は重複とみなす
                if pos_distance < 500 and time_diff < 300:  # 500m以内、5分以内
                    # VMG利得の高い方を保持
                    if point.vmg_gain > existing.vmg_gain:
                        # 既存のポイントを置き換え
                        filtered.remove(existing)
                        filtered.append(point)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered.append(point)
        
        return filtered
    
    def _filter_duplicate_laylines(self, layline_points):
        """重複するレイラインポイントを除去"""
        if len(layline_points) <= 1:
            return layline_points
        
        # 時間でソート
        sorted_points = sorted(layline_points, key=lambda p: p.time_estimate)
        
        # フィルタリング済みのリスト
        filtered = []
        
        for i, point in enumerate(sorted_points):
            # 既存のポイントとの類似性をチェック
            is_duplicate = False
            
            for existing in filtered:
                # 位置の近さをチェック
                pos_distance = self._calculate_distance(
                    point.position[0], point.position[1],
                    existing.position[0], existing.position[1]
                )
                
                # マークまでの距離の類似性
                mark_distance_diff = abs(point.mark_distance - existing.mark_distance)
                mark_distance_similar = mark_distance_diff < 200  # 200m以内なら類似
                
                # 時間の近さをチェック
                time_diff = abs(point.time_estimate - existing.time_estimate)
                
                # 位置と時間が両方近い場合は重複とみなす
                if pos_distance < 300 and time_diff < 300 and mark_distance_similar:  # 300m以内、5分以内
                    # 信頼度の高い方を保持
                    if point.confidence > existing.confidence:
                        # 既存のポイントを置き換え
                        filtered.remove(existing)
                        filtered.append(point)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered.append(point)
        
        return filtered
    
    def _identify_important_legs(self, course_data):
        """
        重要なレグを特定
        
        Parameters:
        -----------
        course_data : Dict
            コースデータ
            
        Returns:
        --------
        List[int]
            重要なレグのインデックスリスト
        """
        important_legs = []
        
        # レグがない場合
        if 'legs' not in course_data or not course_data['legs']:
            return important_legs
        
        # 各レグの重要度を評価
        for i, leg in enumerate(course_data['legs']):
            # 風上レグは重要
            if leg.get('is_upwind', False):
                important_legs.append(i)
                continue
                
            # レグの長さが特に長いものは重要
            path = leg.get('path', {})
            path_points = path.get('path_points', [])
            if len(path_points) > 100:  # 多数のポイントがあるレグは重要
                important_legs.append(i)
                continue
                
            # タックポイントが多いレグは重要
            tack_points = path.get('tack_points', [])
            if len(tack_points) > 2:  # 多数のタックがあるレグは重要
                important_legs.append(i)
                continue
                
            # 最初と最後のレグも重要
            if i == 0 or i == len(course_data['legs']) - 1:
                important_legs.append(i)
        
        return important_legs
    
    def _detect_wind_shifts_for_leg(self, leg, wind_field, target_time, detail_level='high'):
        """特定レグの風向シフトを検出 (最適化版)"""
        # 実装は省略（通常の風向シフト検出と同様）
        return []
    
    def _detect_tacks_for_leg(self, leg, wind_field, detail_level='high'):
        """特定レグのタックを検出 (最適化版)"""
        # 実装は省略（通常のタック検出と同様）
        return []
    
    def _detect_laylines_for_leg(self, leg, wind_field, detail_level='high'):
        """特定レグのレイラインを検出 (最適化版)"""
        # 実装は省略（通常のレイライン検出と同様）
        return []
    
    @lru_cache(maxsize=128)
    def _get_wind_at_position(self, lat: float, lon: float, time_point: Union[datetime, float, None], 
                            wind_field: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """
        指定位置・時間の風情報を取得
        
        Parameters:
        -----------
        lat, lon : float
            位置の緯度・経度
        time_point : datetime, float, or None
            時間点（datetime、タイムスタンプ、またはNone）
        wind_field : Dict[str, Any]
            風の場データ
            
        Returns:
        --------
        Dict[str, float] or None
            風情報（方向、速度、信頼度）
        """
        # WindFieldInterpolatorがあれば利用
        if self.wind_field_interpolator and time_point is not None:
            try:
                # 時間補間を利用した風の場データ取得
                interpolated_field = self.wind_field_interpolator.interpolate_wind_field(
                    target_time=time_point,
                    resolution=None,  # デフォルト解像度
                    method='gp'       # ガウス過程補間
                )
                
                if interpolated_field:
                    return self._extract_wind_at_point(lat, lon, interpolated_field)
            except Exception as e:
                warnings.warn(f"風の場補間エラー: {e}")
        
        # 補間器がないか、失敗した場合は直接風の場から抽出
        return self._extract_wind_at_point(lat, lon, wind_field)
    
    def _extract_wind_at_point(self, lat: float, lon: float, wind_field: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """
        風の場データから特定地点の風情報を抽出
        
        Parameters:
        -----------
        lat, lon : float
            位置の緯度・経度
        wind_field : Dict[str, Any]
            風の場データ
            
        Returns:
        --------
        Dict[str, float] or None
            風情報（方向、速度、信頼度）
        """
        try:
            # グリッドデータを取得
            lat_grid = wind_field['lat_grid']
            lon_grid = wind_field['lon_grid']
            wind_directions = wind_field['wind_direction']
            wind_speeds = wind_field['wind_speed']
            confidence = wind_field.get('confidence', np.ones_like(wind_directions) * 0.8)
            
            # グリッド範囲外の場合
            if (lat < np.min(lat_grid) or lat > np.max(lat_grid) or
                lon < np.min(lon_grid) or lon > np.max(lon_grid)):
                return None
            
            # NumPyベクトル化による最近傍点検索
            distances = (lat_grid - lat)**2 + (lon_grid - lon)**2
            closest_idx = np.unravel_index(np.argmin(distances), distances.shape)
            
            # そのポイントの風データを取得
            direction = float(wind_directions[closest_idx])
            speed = float(wind_speeds[closest_idx])
            conf = float(confidence[closest_idx])
            
            # 風の変動性を計算（近傍9点での標準偏差）
            variability = self._calculate_wind_variability(closest_idx, wind_directions, wind_speeds)
            
            return {
                'direction': direction,
                'speed': speed,
                'confidence': conf,
                'variability': variability
            }
            
        except Exception as e:
            warnings.warn(f"風データ抽出エラー: {e}")
            return None
            
    def _calculate_wind_variability(self, center_idx: Tuple[int, int], 
                                  wind_directions: np.ndarray, 
                                  wind_speeds: np.ndarray) -> float:
        """
        特定地点周辺の風の変動性を計算
        
        Parameters:
        -----------
        center_idx : Tuple[int, int]
            中心点のインデックス
        wind_directions : np.ndarray
            風向データ配列
        wind_speeds : np.ndarray
            風速データ配列
            
        Returns:
        --------
        float
            風の変動性（0-1のスケール）
        """
        # グリッドサイズを取得
        grid_shape = wind_directions.shape
        i, j = center_idx
        
        # 近傍9点の範囲を定義
        i_range = range(max(0, i-1), min(grid_shape[0], i+2))
        j_range = range(max(0, j-1), min(grid_shape[1], j+2))
        
        # 近傍の風向風速を収集
        nearby_dirs = []
        nearby_speeds = []
        
        for ni in i_range:
            for nj in j_range:
                nearby_dirs.append(wind_directions[ni, nj])
                nearby_speeds.append(wind_speeds[ni, nj])
        
        # 風向の変動性（角度データなので特殊処理）
        dir_sin = np.sin(np.radians(nearby_dirs))
        dir_cos = np.cos(np.radians(nearby_dirs))
        
        # 平均ベクトルの長さを算出
        mean_sin = np.mean(dir_sin)
        mean_cos = np.mean(dir_cos)
        r = np.sqrt(mean_sin**2 + mean_cos**2)
        
        # r = 1 は完全に一定、r = 0 は完全にランダム
        dir_variability = 1.0 - r
        
        # 風速の変動性（変動係数 = 標準偏差/平均）
        speed_std = np.std(nearby_speeds)
        speed_mean = np.mean(nearby_speeds)
        
        if speed_mean > 0:
            speed_variability = speed_std / speed_mean
        else:
            speed_variability = 0.0
        
        # 総合変動性（風向の変動が主要因）
        variability = 0.7 * dir_variability + 0.3 * min(1.0, speed_variability)
        
        return min(1.0, max(0.0, variability))
    
    def _is_favorable_shift(self, shift_angle: float, current_course: float, is_upwind: bool) -> bool:
        """
        風向シフトが有利かどうかを判定
        
        Parameters:
        -----------
        shift_angle : float
            風向シフト角度（度、正：時計回り、負：反時計回り）
        current_course : float
            現在のコース（度）
        is_upwind : bool
            風上レグかどうか
            
        Returns:
        --------
        bool
            有利なシフトならTrue
        """
        # 風上と風下では評価が逆になる
        if is_upwind:
            # 風上向かいレグ - タックしないで近づけるシフトが有利
            relative_angle = angle_difference(current_course, shift_angle)
            return relative_angle > 0  # 風が艇の方向に向かってくるシフト
        else:
            # 風下レグ - 艇から離れるシフトが有利
            relative_angle = angle_difference(current_course, shift_angle)
            return relative_angle < 0  # 風が艇から離れるシフト
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float
