"""
セーリング戦略分析システム - 戦略ポイント可視化モジュール

検出された戦略的判断ポイントを視覚化し、
セーラーの意思決定をサポートするための可視化機能を提供します。
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from typing import List, Dict, Tuple, Optional, Union, Any
import warnings
import math

class StrategyVisualizer:
    """戦略ポイントの可視化を担当するクラス"""
    
    def __init__(self):
        """初期化"""
        # 可視化設定
        self.config = {
            'figure_size': (12, 10),         # プロット図のサイズ
            'dpi': 200,                      # 解像度
            'grid_alpha': 0.7,               # グリッドの透明度
            'point_size_base': 100,          # 基本ポイントサイズ
            'point_size_scale': 1.5,         # 重要度によるサイズスケール
            'wind_arrow_scale': 0.0002,      # 風向矢印のスケール
            'wind_arrow_alpha': 0.7,         # 風向矢印の透明度
            'course_line_width': 1.5,        # コース線の幅
            'course_line_alpha': 0.5,        # コース線の透明度
            'mark_size': 150,                # マークサイズ
            'label_fontsize': 10,            # ラベルのフォントサイズ
            'title_fontsize': 14,            # タイトルのフォントサイズ
            'show_point_labels': True,       # ポイントラベルの表示
            'show_wind_arrows': True,        # 風向矢印の表示
            'show_confidence': True,         # 信頼度（透明度）の表示
            'max_points': 20,                # 表示する最大ポイント数
        }
        
        # ポイント種類ごとの表示スタイル
        self.point_styles = {
            "tack": {
                'marker': '^',               # 三角形
                'color': '#2196F3',          # 青
                'label': 'Tack/Gybe Point',
                'zorder': 10                 # 描画順序（大きいほど前面）
            },
            "wind_shift": {
                'marker': 'o',               # 円
                'favorable_color': '#4CAF50', # 有利なシフトは緑
                'unfavorable_color': '#F44336', # 不利なシフトは赤
                'label': 'Wind Shift Point',
                'zorder': 11
            },
            "layline": {
                'marker': 'X',               # バツ印
                'color': '#9C27B0',          # 紫
                'label': 'Layline Point',
                'zorder': 12
            }
        }
    
    def update_config(self, **kwargs):
        """
        可視化設定を更新
        
        Parameters:
        -----------
        **kwargs
            更新する設定のキーと値
        """
        for key, value in kwargs.items():
            if key in self.config:
                self.config[key] = value
            else:
                warnings.warn(f"未知の設定キー: {key}")
    
    def visualize_strategy_points(self, strategy_points: List[Any], course_data: Optional[Dict[str, Any]] = None, 
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        戦略ポイントを可視化
        
        Parameters:
        -----------
        strategy_points : List[StrategyPoint]
            可視化する戦略ポイント
        course_data : Dict[str, Any], optional
            コースデータ（レグ、マークなどの情報）
        save_path : str, optional
            保存先パス
            
        Returns:
        --------
        plt.Figure
            プロット図
        """
        if not strategy_points:
            warnings.warn("可視化するポイントがありません")
            return None
        
        # プロット設定
        fig, ax = plt.subplots(figsize=self.config['figure_size'])
        
        # 重要度でソートして上限数に制限
        sorted_points = sorted(strategy_points, key=lambda p: getattr(p, 'importance', 0), reverse=True)
        points_to_show = sorted_points[:self.config['max_points']]
        
        # コースデータがあれば表示
        if course_data:
            self._plot_course(ax, course_data)
        
        # 点のプロット（種類ごとに一度だけラベル付け）
        labeled_types = set()
        
        for i, point in enumerate(points_to_show):
            point_type = getattr(point, 'point_type', 'unknown')
            lat, lon = point.position if hasattr(point, 'position') else (0, 0)
            
            # ポイントスタイルの取得
            style = self.point_styles.get(point_type, {
                'marker': 'x', 
                'color': 'black', 
                'label': 'Unknown Point',
                'zorder': 9
            })
            
            # 重要度に基づくサイズ調整
            importance = getattr(point, 'importance', 0.5)
            size = self.config['point_size_base'] * (0.5 + importance * self.config['point_size_scale'])
            
            # 信頼度に基づく透明度
            confidence = getattr(point, 'confidence', 0.8)
            alpha = 0.4 + confidence * 0.6 if self.config['show_confidence'] else 1.0
            
            # マーカーの色を決定
            if point_type == "wind_shift" and hasattr(point, 'favorable'):
                # 風向シフトは有利/不利で色分け
                marker_color = style['favorable_color'] if point.favorable else style['unfavorable_color']
            else:
                marker_color = style.get('color', 'black')
            
            # ラベル付け（各種類一度だけ）
            label = style['label'] if point_type not in labeled_types else None
            if label:
                labeled_types.add(point_type)
            
            # 点のプロット
            ax.scatter(lon, lat, 
                     marker=style['marker'], 
                     c=marker_color, 
                     s=size, 
                     alpha=alpha, 
                     label=label,
                     zorder=style['zorder'])
            
            # ポイント番号を表示
            if self.config['show_point_labels']:
                ax.text(lon, lat, str(i + 1), 
                      fontsize=self.config['label_fontsize'], 
                      ha='center', va='center', 
                      color='white', fontweight='bold',
                      zorder=style['zorder'] + 1)
            
            # 風向を矢印で表示
            if self.config['show_wind_arrows'] and hasattr(point, 'wind_info'):
                self._plot_wind_arrow(ax, point)
        
        # レジェンド表示
        ax.legend(loc='upper left')
        
        # グリッドと軸ラベル
        ax.grid(True, linestyle='--', alpha=self.config['grid_alpha'])
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Strategic Decision Points', fontsize=self.config['title_fontsize'])
        
        # アスペクト比調整
        ax.set_aspect('equal')
        
        # 保存処理
        if save_path:
            plt.savefig(save_path, dpi=self.config['dpi'], bbox_inches='tight')
        
        return fig
    
    def create_strategy_summary(self, strategy_points: List[Any], 
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        戦略ポイントのサマリー表示（テキスト+グラフィック）
        
        Parameters:
        -----------
        strategy_points : List[StrategyPoint]
            サマリー表示する戦略ポイント
        save_path : str, optional
            保存先パス
            
        Returns:
        --------
        plt.Figure
            サマリー図
        """
        if not strategy_points:
            warnings.warn("サマリー表示するポイントがありません")
            return None
        
        # 重要度でソートして上限数に制限
        sorted_points = sorted(strategy_points, key=lambda p: getattr(p, 'importance', 0), reverse=True)
        points_to_show = sorted_points[:min(5, len(sorted_points))]  # 最大5ポイント
        
        # プロット設定
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # 背景を透明に
        ax.set_axis_off()
        
        # サマリータイトル
        plt.text(0.5, 0.95, 'Strategic Decision Points Summary', 
                ha='center', va='top', fontsize=16, fontweight='bold')
        
        y_pos = 0.9
        line_height = 0.15
        
        # 各ポイントの情報表示
        for i, point in enumerate(points_to_show):
            point_type = getattr(point, 'point_type', 'unknown')
            
            # ポイントスタイルの取得
            style = self.point_styles.get(point_type, {
                'marker': 'x', 
                'color': 'black', 
                'label': 'Unknown Point'
            })
            
            # マーカーの色を決定
            if point_type == "wind_shift" and hasattr(point, 'favorable'):
                marker_color = style['favorable_color'] if point.favorable else style['unfavorable_color']
            else:
                marker_color = style.get('color', 'black')
            
            # ポイント情報
            importance = getattr(point, 'importance', 0.5)
            confidence = getattr(point, 'confidence', 0.8)
            
            # サマリーテキスト
            header_text = f"#{i+1}: {style['label']} (Importance: {importance:.2f}, Confidence: {confidence:.2f})"
            
            # ポイント番号と種類
            plt.text(0.05, y_pos, header_text, va='top', fontsize=12, fontweight='bold')
            
            # 説明文
            description = getattr(point, 'description', 'No description available.')
            plt.text(0.07, y_pos - 0.04, description, va='top', fontsize=10)
            
            # 推奨アクション
            recommendation = getattr(point, 'recommendation', '')
            if recommendation:
                plt.text(0.07, y_pos - 0.08, f"Recommended Action: {recommendation}", 
                       va='top', fontsize=10, color='#1976D2')
            
            # リスクスコア表示（グラフィカル）
            risk_score = getattr(point, 'risk_score', 50)
            risk_color = self._get_risk_color(risk_score)
            plt.text(0.85, y_pos - 0.04, f"Risk: {risk_score:.0f}", va='top', fontsize=10)
            
            # リスクバー
            bar_width = 0.1
            bar_height = 0.01
            ax.add_patch(plt.Rectangle((0.85, y_pos - 0.07), bar_width, bar_height, 
                                     color='#E0E0E0'))
            ax.add_patch(plt.Rectangle((0.85, y_pos - 0.07), bar_width * (risk_score / 100), 
                                     bar_height, color=risk_color))
            
            # マーカー表示
            marker_size = 100
            ax.scatter(0.03, y_pos - 0.03, marker=style['marker'], c=marker_color, 
                     s=marker_size, alpha=0.8)
            
            # 次のポイント位置
            y_pos -= line_height
        
        # 凡例の作成
        legend_elements = [
            Patch(facecolor='#4CAF50', label='Favorable Wind Shift'),
            Patch(facecolor='#F44336', label='Unfavorable Wind Shift'),
            Patch(facecolor='#2196F3', label='Tack/Gybe Point'),
            Patch(facecolor='#9C27B0', label='Layline Point')
        ]
        
        plt.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.05), 
                 ncol=4, fancybox=True, shadow=True)
        
        # 保存処理
        if save_path:
            plt.savefig(save_path, dpi=self.config['dpi'], bbox_inches='tight')
        
        return fig
    
    def create_time_sequence_view(self, strategy_points: List[Any], current_time: float = 0, 
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        時間順に戦略ポイントを可視化
        
        Parameters:
        -----------
        strategy_points : List[StrategyPoint]
            可視化する戦略ポイント
        current_time : float
            現在時刻
        save_path : str, optional
            保存先パス
            
        Returns:
        --------
        plt.Figure
            タイムライン図
        """
        if not strategy_points:
            warnings.warn("可視化するポイントがありません")
            return None
        
        # 時間でソート
        sorted_points = sorted(strategy_points, key=lambda p: getattr(p, 'time_estimate', 0))
        
        # プロット設定
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 背景とグリッド
        ax.set_facecolor('#F5F5F5')
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # 時間範囲の取得
        min_time = min([getattr(p, 'time_estimate', 0) for p in sorted_points])
        max_time = max([getattr(p, 'time_estimate', 3600) for p in sorted_points])
        
        # 余裕を持たせた時間範囲
        time_range = max_time - min_time
        padding = time_range * 0.1
        ax.set_xlim(min_time - padding, max_time + padding)
        
        # Y軸方向の位置調整用
        y_positions = {}
        y_offset = 0
        
        # 現在時刻の垂直線
        if current_time > 0:
            ax.axvline(x=current_time, color='red', linestyle='-', linewidth=2, alpha=0.7)
            ax.text(current_time, -0.05, 'Now', ha='center', va='top', 
                  transform=ax.get_xaxis_transform(), color='red')
        
        # 各ポイントをプロット
        for i, point in enumerate(sorted_points):
            point_type = getattr(point, 'point_type', 'unknown')
            time = getattr(point, 'time_estimate', 0)
            
            # ポイントスタイルの取得
            style = self.point_styles.get(point_type, {
                'marker': 'x', 
                'color': 'black', 
                'label': 'Unknown Point'
            })
            
            # マーカーの色を決定
            if point_type == "wind_shift" and hasattr(point, 'favorable'):
                marker_color = style['favorable_color'] if point.favorable else style['unfavorable_color']
            else:
                marker_color = style.get('color', 'black')
            
            # Y位置の計算（オーバーラップを避けるため）
            y_pos = self._calculate_y_position(time, y_positions, 0.05)
            y_positions[time] = y_pos
            
            # ポイントプロット
            importance = getattr(point, 'importance', 0.5)
            marker_size = 100 * (0.5 + importance)
            ax.scatter(time, y_pos, marker=style['marker'], c=marker_color, 
                     s=marker_size, alpha=0.8)
            
            # ラベル
            description = getattr(point, 'description', f"{style['label']}")
            short_desc = self._shorten_description(description, 40)
            
            # 時間の表現
            if time > current_time:
                time_text = f"+{((time - current_time) / 60):.1f}min"
            else:
                time_text = f"{time:.0f}s"
            
            # テキスト配置
            ax.text(time, y_pos + 0.02, f"#{i+1} {short_desc}",
                  ha='left', va='center', fontsize=9, rotation=0)
            
            # 時間表示
            ax.text(time, y_pos - 0.02, time_text,
                  ha='left', va='center', fontsize=8, rotation=0,
                  color='gray')
        
        # 軸ラベル設定
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('')
        ax.set_yticks([])
        ax.set_title('Strategic Decision Points Timeline', fontsize=14)
        
        # X軸を時間単位で表示
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/60:.0f}min"))
        
        # 保存処理
        if save_path:
            plt.savefig(save_path, dpi=self.config['dpi'], bbox_inches='tight')
        
        return fig
    
    def create_risk_vs_gain_plot(self, strategy_points: List[Any], 
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        リスク対利得のプロット
        
        Parameters:
        -----------
        strategy_points : List[StrategyPoint]
            分析する戦略ポイント
        save_path : str, optional
            保存先パス
            
        Returns:
        --------
        plt.Figure
            リスク対利得プロット
        """
        if not strategy_points:
            warnings.warn("分析するポイントがありません")
            return None
        
        # プロット設定
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 各ポイント種類ごとにデータを整理
        point_data = {}
        
        for point_type in ['wind_shift', 'tack', 'layline']:
            point_data[point_type] = {
                'risks': [],
                'gains': [],
                'importance': [],
                'confidence': [],
                'is_favorable': [],
                'points': []
            }
        
        # 各ポイントのデータを収集
        for point in strategy_points:
            point_type = getattr(point, 'point_type', 'unknown')
            if point_type not in point_data:
                continue
                
            # リスクスコア
            risk = getattr(point, 'risk_score', 50)
            
            # 利得計算（ポイント種類に応じて）
            gain = 0
            if point_type == 'wind_shift':
                # 風向シフトの場合
                shift_angle = getattr(point, 'shift_angle', 0)
                favorable = getattr(point, 'favorable', False)
                gain = abs(shift_angle) / 45.0  # 45度を1.0とする
                if favorable:
                    gain *= 1.5  # 有利なシフトは価値が高い
            elif point_type == 'tack':
                # タックポイントの場合
                vmg_gain = getattr(point, 'vmg_gain', 0)
                gain = vmg_gain * 5  # VMG利得20%を1.0とする
            elif point_type == 'layline':
                # レイラインの場合
                mark_distance = getattr(point, 'mark_distance', 1000)
                gain = 1.0 - min(1.0, mark_distance / 2000)  # 距離に反比例
            
            # 信頼度とバブルサイズ
            importance = getattr(point, 'importance', 0.5)
            confidence = getattr(point, 'confidence', 0.8)
            
            # 各種類のデータに追加
            point_data[point_type]['risks'].append(risk)
            point_data[point_type]['gains'].append(gain)
            point_data[point_type]['importance'].append(importance)
            point_data[point_type]['confidence'].append(confidence)
            point_data[point_type]['is_favorable'].append(getattr(point, 'favorable', None))
            point_data[point_type]['points'].append(point)
        
        # 各ポイント種類ごとにプロット
        for point_type, data in point_data.items():
            if not data['risks']:
                continue
                
            # スタイル取得
            style = self.point_styles.get(point_type, {
                'marker': 'x', 
                'color': 'black', 
                'label': 'Unknown Point'
            })
            
            # 色設定
            if point_type == 'wind_shift':
                # 風向シフトは有利/不利で色分け
                colors = []
                for is_favorable in data['is_favorable']:
                    if is_favorable is None:
                        colors.append(style.get('color', 'gray'))
                    elif is_favorable:
                        colors.append(style.get('favorable_color', 'green'))
                    else:
                        colors.append(style.get('unfavorable_color', 'red'))
            else:
                colors = [style.get('color', 'black')] * len(data['risks'])
            
            # バブルサイズ（重要度と信頼度で調整）
            sizes = [self.config['point_size_base'] * (0.5 + imp) * conf 
                   for imp, conf in zip(data['importance'], data['confidence'])]
            
            # プロット
            scatter = ax.scatter(data['risks'], data['gains'], 
                               s=sizes, 
                               c=colors,
                               marker=style.get('marker', 'o'),
                               alpha=0.7,
                               label=style.get('label', point_type))
            
            # ポイントラベル
            for i, point in enumerate(data['points']):
                ax.text(data['risks'][i], data['gains'][i], 
                      str(strategy_points.index(point) + 1), 
                      ha='center', va='center', 
                      color='white', fontweight='bold',
                      fontsize=8)
        
        # 象限ラベル
        ax.text(75, 0.75, "High Risk\nHigh Gain", ha='center', va='center', 
              fontsize=10, alpha=0.7, bbox=dict(facecolor='white', alpha=0.5))
        ax.text(25, 0.75, "Low Risk\nHigh Gain", ha='center', va='center', 
              fontsize=10, alpha=0.7, bbox=dict(facecolor='white', alpha=0.5))
        ax.text(75, 0.25, "High Risk\nLow Gain", ha='center', va='center', 
              fontsize=10, alpha=0.7, bbox=dict(facecolor='white', alpha=0.5))
        ax.text(25, 0.25, "Low Risk\nLow Gain", ha='center', va='center', 
              fontsize=10, alpha=0.7, bbox=dict(facecolor='white', alpha=0.5))
        
        # 軸設定
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel('Risk Score', fontsize=12)
        ax.set_ylabel('Potential Gain', fontsize=12)
        ax.set_title('Risk vs. Gain Analysis', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 中央十字線
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=50, color='gray', linestyle='--', alpha=0.5)
        
        # 凡例
        ax.legend(loc='upper left')
        
        # 保存処理
        if save_path:
            plt.savefig(save_path, dpi=self.config['dpi'], bbox_inches='tight')
        
        return fig
    
    def _plot_course(self, ax, course_data):
        """コースデータを描画"""
        if 'legs' not in course_data:
            return
            
        legs = course_data.get('legs', [])
        waypoints = course_data.get('waypoints', [])
        
        # コースレグを描画
        for leg in legs:
            # レグタイプによって色を変更
            color = 'red' if leg.get('is_upwind', False) else 'blue'
            
            # パスポイントがあれば描画
            path = leg.get('path', {})
            if 'path_points' in path:
                path_points = path['path_points']
                lats = [p.get('lat') for p in path_points if 'lat' in p]
                lons = [p.get('lon') for p in path_points if 'lon' in p]
                
                if lats and lons:
                    ax.plot(lons, lats, color=color, linewidth=self.config['course_line_width'], 
                          alpha=self.config['course_line_alpha'])
        
        # ウェイポイント（マーク）を描画
        for wp in waypoints:
            if 'lat' in wp and 'lon' in wp:
                ax.scatter(wp['lon'], wp['lat'], 
                         marker='o', 
                         c='black', 
                         s=self.config['mark_size'], 
                         alpha=0.7,
                         zorder=5)
                
                # マーク名を表示
                if 'name' in wp:
                    ax.text(wp['lon'], wp['lat'], wp['name'], 
                          fontsize=self.config['label_fontsize'],
                          ha='center', va='bottom',
                          color='black',
                          bbox=dict(facecolor='white', alpha=0.7))
    
    def _plot_wind_arrow(self, ax, point):
        """風向矢印を描画"""
        if not hasattr(point, 'wind_info'):
            return
            
        wind_info = point.wind_info
        wind_dir = wind_info.get('direction', 0)
        wind_speed = wind_info.get('speed', 0)
        
        # 位置
        lat, lon = point.position
        
        # 風向を矢印で表示（風が吹いてくる方向）
        arrow_length = self.config['wind_arrow_scale'] * wind_speed  # 風速に応じた矢印の長さ
        
        # 風向は気象学的（風が吹いてくる方向）なので、ベクトルは逆向き
        dx = -arrow_length * np.sin(np.radians(wind_dir))
        dy = -arrow_length * np.cos(np.radians(wind_dir))
        
        # 矢印描画
        ax.arrow(lon, lat, dx, dy, 
               head_width=arrow_length * 0.5, 
               head_length=arrow_length * 0.8, 
               fc='gray', ec='gray', 
               alpha=self.config['wind_arrow_alpha'],
               zorder=5)
    
    def _get_risk_color(self, risk_score):
        """リスクスコアに基づく色を返す"""
        if risk_score < 25:
            return '#4CAF50'  # 低リスク: 緑
        elif risk_score < 50:
            return '#FFEB3B'  # 中低リスク: 黄
        elif risk_score < 75:
            return '#FF9800'  # 中高リスク: オレンジ
        else:
            return '#F44336'  # 高リスク: 赤
    
    def _shorten_description(self, description, max_length=40):
        """説明文を短縮"""
        if len(description) <= max_length:
            return description
        return description[:max_length - 3] + '...'
    
    def _calculate_y_position(self, time, existing_positions, min_spacing):
        """タイムライン表示での重複を避けるY座標を計算"""
        position = 0.5  # デフォルト中央
        
        # 近い時間のポイントを探す
        nearby_positions = []
        for t, pos in existing_positions.items():
            if abs(t - time) < 120:  # 2分以内は近いと判断
                nearby_positions.append(pos)
        
        # 近くに他のポイントがある場合
        if nearby_positions:
            # すでに使われているY位置を避ける
            while any(abs(position - pos) < min_spacing for pos in nearby_positions):
                # 少しずつずらす（上下交互）
                offset = min_spacing * (1 + len(nearby_positions) % 2)
                position += offset * (1 if len(nearby_positions) % 2 == 0 else -1)
                
                # 範囲制限
                position = max(0.1, min(0.9, position))
        
        return position
