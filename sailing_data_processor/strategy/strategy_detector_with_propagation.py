"""
StrategyDetectorWithPropagation 

�n4n�Ո,�nW_&e�h����
"""

import numpy as np
import math
import warnings
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, timedelta
from functools import lru_cache

# ������n�����
from sailing_data_processor.strategy.detector import StrategyDetector
from sailing_data_processor.strategy.points import StrategyPoint, WindShiftPoint, TackPoint, LaylinePoint

# ���-�
logger = logging.getLogger(__name__)

class StrategyDetectorWithPropagation(StrategyDetector):
    """
    �n4n�Ո,�nW_&e�h
    
    StrategyDetectorn_�k�Hf�n4n
    e�j�ՒnW~Y
    """
    
    def __init__(self, vmg_calculator=None, wind_fusion_system=None):
        """
        
        
        Parameters:
        -----------
        vmg_calculator : OptimalVMGCalculator, optional
            VMG�h
        wind_fusion_system : WindFieldFusionSystem, optional
            �q����
        """
        # ���
        super().__init__(vmg_calculator)
        
        # �q����-�
        self.wind_fusion_system = wind_fusion_system
        
        # �Ո,-�
        self.propagation_config = {
            'wind_shift_prediction_horizon': 1800,  # �	�,B��	
            'prediction_time_step': 300,           # �,�����	
            'wind_shift_confidence_threshold': 0.7,  # �	�<��$
            'min_propagation_distance': 1000,      #  �������	
            'prediction_confidence_decay': 0.1,    # �,�<�nB�p
            'use_historical_data': True            # N����n)(�&
        }
        
        logger.info("StrategyDetectorWithPropagation initialized")
    
    def detect_wind_shifts_with_propagation(self, course_data: Dict[str, Any], 
                                          wind_field: Dict[str, Any]) -> List[WindShiftPoint]:
        """
        ��,�nW_�	�
        
        �n4nB�zU�nWen�	��,
        �9$��gn�	��Y�
        
        Parameters:
        -----------
        course_data : dict
            ������
        wind_field : dict
            �n4���
            
        Returns:
        --------
        List[WindShiftPoint]
            �U�_�	�n��
        """
        logger.debug("Starting wind shift detection with propagation")
        
        wind_shifts = []
        
        # �q����LjD4o8n�
        if not self.wind_fusion_system:
            logger.warning("No wind fusion system available. Using standard detection.")
            return self._detect_wind_shifts_in_legs(course_data, wind_field)
        
        try:
            # ������K�B���֗
            timestamps = course_data.get('timestamps', [])
            if not timestamps:
                logger.warning("No timestamps in course data")
                return []
            
            start_time = min(timestamps)
            end_time = max(timestamps)
            
            # �,B�����g�
            current_time = start_time
            while current_time <= end_time:
                try:
                    # �B;gn�n4�֗
                    current_wind_field = self._get_wind_field_at_time(current_time, wind_field)
                    
                    if current_wind_field:
                        # �gn�	��
                        shifts_at_time = self._detect_wind_shifts_in_legs(
                            course_data, current_wind_field, target_time=current_time
                        )
                        
                        # �U�_	����
                        wind_shifts.extend(shifts_at_time)
                    
                except Exception as e:
                    logger.error(f"Error processing time {current_time}: {e}")
                
                # !nB�����x
                current_time += timedelta(seconds=self.propagation_config['prediction_time_step'])
            
            # �nd�h�<�k��^��
            wind_shifts = self._filter_wind_shifts(wind_shifts)
            
            logger.info(f"Detected {len(wind_shifts)} wind shifts with propagation")
            
        except Exception as e:
            logger.error(f"Wind shift detection with propagation failed: {e}")
            # թ���ïhWf8n��(
            wind_shifts = self._detect_wind_shifts_in_legs(course_data, wind_field)
        
        return wind_shifts
    
    def _get_wind_field_at_time(self, target_time: Union[datetime, float], 
                               wind_field: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        �B;gn�n4�֗
        
        �q�����(WfB�ܓU�_�n4�֗
        
        Parameters:
        -----------
        target_time : datetime or float
            �aB;
        wind_field : dict
            ��hj��n4���
            
        Returns:
        --------
        dict or None
            B�ܓU�_�n4���
        """
        try:
            if self.wind_fusion_system and hasattr(self.wind_fusion_system, 'interpolate_wind_field'):
                # �q����nB�ܓ��ɒ(
                interpolated_field = self.wind_fusion_system.interpolate_wind_field(
                    target_time=target_time,
                    method='linear'  # ~_oGPRBFji
                )
                return interpolated_field
            else:
                # �q����L)(gMjD4oCn�����Y
                return wind_field
                
        except Exception as e:
            logger.error(f"Failed to get wind field at time {target_time}: {e}")
            return None
    
    def _detect_wind_shifts_in_legs(self, course_data: Dict[str, Any], 
                                   wind_field: Dict[str, Any],
                                   target_time: Optional[Union[datetime, float]] = None) -> List[WindShiftPoint]:
        """
        �gn�	�
        
        ���n�g�	��Y�
        
        Parameters:
        -----------
        course_data : dict
            ������
        wind_field : dict
            �n4���
        target_time : datetime or float, optional
            �aB;
            
        Returns:
        --------
        List[WindShiftPoint]
            �U�_�	�n��
        """
        wind_shifts = []
        
        try:
            # ������n֗
            positions = course_data.get('positions', [])
            timestamps = course_data.get('timestamps', [])
            
            if not positions or not timestamps:
                logger.warning("Insufficient data for wind shift detection")
                return []
            
            # B�gn�
            for i in range(1, len(positions)):
                # �(hMnMn
                prev_pos = positions[i-1]
                curr_pos = positions[i]
                
                # ��๿��
                prev_time = timestamps[i-1]
                curr_time = timestamps[i]
                
                # �aB;n�LB�4o����
                if target_time and abs((curr_time - target_time).total_seconds()) > self.propagation_config['prediction_time_step']:
                    continue
                
                # Mngn��֗
                prev_wind = self._get_wind_at_position(
                    prev_pos['lat'], prev_pos['lon'], prev_time, wind_field
                )
                curr_wind = self._get_wind_at_position(
                    curr_pos['lat'], curr_pos['lon'], curr_time, wind_field
                )
                
                if not prev_wind or not curr_wind:
                    continue
                
                # �	��
                wind_change = self._calculate_wind_direction_change(
                    prev_wind['direction'], curr_wind['direction']
                )
                
                # �$��H�	��
                if abs(wind_change) > self.config.get('min_wind_shift_angle', 5.0):
                    confidence = self._calculate_wind_shift_confidence(
                        wind_change, prev_wind.get('confidence', 0.7), 
                        curr_wind.get('confidence', 0.7)
                    )
                    
                    wind_shift = WindShiftPoint(
                        position=curr_pos,
                        timestamp=curr_time,
                        wind_data={
                            'direction_before': prev_wind['direction'],
                            'direction_after': curr_wind['direction'],
                            'direction_change': wind_change,
                            'speed_before': prev_wind.get('speed', 0),
                            'speed_after': curr_wind.get('speed', 0)
                        },
                        confidence=confidence,
                        magnitude=abs(wind_change)
                    )
                    
                    wind_shifts.append(wind_shift)
                    
        except Exception as e:
            logger.error(f"Error in wind shift detection in legs: {e}")
        
        return wind_shifts
    
    def _calculate_wind_direction_change(self, dir1: float, dir2: float) -> float:
        """
        �	��-180^180�n��	
        
        Parameters:
        -----------
        dir1 : float
            �˨�	
        dir2 : float
            B���	
            
        Returns:
        --------
        float
            �	�c=Bފ�=�Bފ	
        """
        # Ҧ��
        diff = dir2 - dir1
        
        # -180^180n��kc�
        if diff > 180:
            diff -= 360
        elif diff < -180:
            diff += 360
            
        return diff
    
    def _calculate_wind_shift_confidence(self, wind_change: float, 
                                       confidence_before: float, 
                                       confidence_after: float) -> float:
        """
        �	n�<���
        
        Parameters:
        -----------
        wind_change : float
            �	��	
        confidence_before : float
            	Mn�����<�
        confidence_after : float
            	�n�����<�
            
        Returns:
        --------
        float
            �	n�<�0-1	
        """
        # �,�<�oM�nsG
        base_confidence = (confidence_before + confidence_after) / 2
        
        # 	�k����Q'Mj	{i�<�N	
        change_weight = 1.0 / (1.0 + abs(wind_change) / 45.0)
        
        return base_confidence * change_weight
    
    def _filter_wind_shifts(self, wind_shifts: List[WindShiftPoint]) -> List[WindShiftPoint]:
        """
        �	�nգ���
        
        �Jd�<�գ���́���ȒLF
        
        Parameters:
        -----------
        wind_shifts : List[WindShiftPoint]
            �U�_�	�n��
            
        Returns:
        --------
        List[WindShiftPoint]
            գ���U�_�	�n��
        """
        if not wind_shifts:
            return []
        
        # �<�gգ���
        min_confidence = self.propagation_config.get('wind_shift_confidence_threshold', 0.7)
        filtered_shifts = [
            shift for shift in wind_shifts if shift.confidence >= min_confidence
        ]
        
        # B��k�D	�����
        merged_shifts = self._merge_close_wind_shifts(filtered_shifts)
        
        # ́�	���<�	g���
        merged_shifts.sort(
            key=lambda x: x.magnitude * x.confidence, 
            reverse=True
        )
        
        return merged_shifts
    
    def _merge_close_wind_shifts(self, wind_shifts: List[WindShiftPoint], 
                                time_window: float = 300) -> List[WindShiftPoint]:
        """
        B��k�D�	�����
        
        Parameters:
        -----------
        wind_shifts : List[WindShiftPoint]
            �	�n��
        time_window : float
            ���Y�B���	
            
        Returns:
        --------
        List[WindShiftPoint]
            ���U�_�	�n��
        """
        if not wind_shifts:
            return []
        
        # B�k���
        sorted_shifts = sorted(wind_shifts, key=lambda x: x.timestamp)
        
        merged = []
        current_group = [sorted_shifts[0]]
        
        for i in range(1, len(sorted_shifts)):
            shift = sorted_shifts[i]
            last_in_group = current_group[-1]
            
            # B���
            time_diff = (shift.timestamp - last_in_group.timestamp).total_seconds()
            
            if time_diff <= time_window:
                # X����k��
                current_group.append(shift)
            else:
                # �(n���ג���Wf��
                merged.append(self._merge_wind_shift_group(current_group))
                current_group = [shift]
        
        #  �n���ג�
        if current_group:
            merged.append(self._merge_wind_shift_group(current_group))
        
        return merged
    
    def _merge_wind_shift_group(self, shifts: List[WindShiftPoint]) -> WindShiftPoint:
        """
        �	����ג���
        
        Parameters:
        -----------
        shifts : List[WindShiftPoint]
            ���Y��	�n����
            
        Returns:
        --------
        WindShiftPoint
            ���U�_�	�
        """
        if len(shifts) == 1:
            return shifts[0]
        
        #  ��<�L�D	��x�
        best_shift = max(shifts, key=lambda x: x.confidence)
        
        # sG�jy'��
        avg_magnitude = np.mean([s.magnitude for s in shifts])
        avg_confidence = np.mean([s.confidence for s in shifts])
        
        # qU�_�	���
        wind_data = best_shift.wind_data.copy()
        wind_data['merged_count'] = len(shifts)
        wind_data['avg_magnitude'] = avg_magnitude
        
        # ���U�_	���Y
        return WindShiftPoint(
            position=best_shift.position,
            timestamp=best_shift.timestamp,
            wind_data=wind_data,
            confidence=avg_confidence,
            magnitude=avg_magnitude
        )
    
    def propagate_wind_field(self, wind_field: Dict[str, Any], 
                           propagation_time: float) -> Dict[str, Any]:
        """
        �n4�B�zUU[�
        
        �n4��B�`QekQf�U[�
        
        Parameters:
        -----------
        wind_field : dict
            �(n�n4���
        propagation_time : float
            �B��	
            
        Returns:
        --------
        dict
            ��n�n4���
        """
        if not self.wind_fusion_system or not hasattr(self.wind_fusion_system, 'propagate_field'):
            logger.warning("No propagation method available in wind fusion system")
            return wind_field
        
        try:
            # �q����n���ɒ(
            propagated_field = self.wind_fusion_system.propagate_field(
                wind_field=wind_field,
                time_delta=propagation_time
            )
            
            # �<��B�pU[�
            if 'confidence' in propagated_field:
                decay_factor = math.exp(
                    -self.propagation_config['prediction_confidence_decay'] * 
                    propagation_time / 3600  # B�XMgp
                )
                propagated_field['confidence'] *= decay_factor
            
            return propagated_field
            
        except Exception as e:
            logger.error(f"Wind field propagation failed: {e}")
            return wind_field
    
    def estimate_future_conditions(self, position: Dict[str, float], 
                                 time_horizon: float) -> List[Dict[str, Any]]:
        """
        en�����
        
        �Mngnen��	��,Y�
        
        Parameters:
        -----------
        position : dict
            �,Mnlat, lon	
        time_horizon : float
            �,B����	
            
        Returns:
        --------
        List[dict]
            B�n��,���
        """
        predictions = []
        
        try:
            # �(n�n4�֗
            current_field = self.wind_fusion_system.get_current_field()
            
            # �,����Thk�
            time_steps = np.arange(
                0, time_horizon, 
                self.propagation_config['prediction_time_step']
            )
            
            for t in time_steps:
                # �n4��
                future_field = self.propagate_wind_field(current_field, t)
                
                # �Mngn��֗
                wind_at_position = self._get_wind_at_position(
                    position['lat'], position['lon'], 
                    datetime.now() + timedelta(seconds=t), 
                    future_field
                )
                
                if wind_at_position:
                    predictions.append({
                        'time_offset': t,
                        'wind_direction': wind_at_position['direction'],
                        'wind_speed': wind_at_position['speed'],
                        'confidence': wind_at_position.get('confidence', 0.7)
                    })
                    
        except Exception as e:
            logger.error(f"Future conditions estimation failed: {e}")
        
        return predictions
    
    def __str__(self) -> str:
        """�Wh�"""
        return f"StrategyDetectorWithPropagation(config={self.propagation_config})"
    
    def __repr__(self) -> str:
        """s0�Wh�"""
        return (f"StrategyDetectorWithPropagation("
                f"vmg_calculator={self.vmg_calculator}, "
                f"wind_fusion_system={self.wind_fusion_system}, "
                f"propagation_config={self.propagation_config})")
