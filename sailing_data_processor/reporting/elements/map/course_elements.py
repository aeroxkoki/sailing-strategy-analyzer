"""
sailing_data_processor.reporting.elements.map.course_elements

���� ����ЛY�����gY
���Mn���b����&eݤ��jin_����W~Y
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import json
import uuid

from sailing_data_processor.reporting.elements.visualizations.map_elements import StrategyPointLayerElement
from sailing_data_processor.reporting.templates.template_model import ElementType, ElementModel


class CourseElementsLayer(StrategyPointLayerElement):
    """
    ���� ���
    
    ������
n�����&eݤ��ji�
    h:Y�_�n���� �ЛW~Y
    """
    
    def __init__(self, model: Optional[ElementModel] = None, **kwargs):
        """
        
        
        Parameters
        ----------
        model : Optional[ElementModel], optional
            � ���, by default None
        **kwargs : dict
            ���LЛU�jD4k(U�����ƣ
        """
        super().__init__(model, **kwargs)
        
        # ����#n-�
        self.set_property("marks", self.get_property("marks", []))
        self.set_property("course_shape", self.get_property("course_shape", "windward_leeward"))
        self.set_property("start_line", self.get_property("start_line", {}))
        self.set_property("finish_line", self.get_property("finish_line", {}))
        
        # ���#n-�
        self.set_property("show_laylines", self.get_property("show_laylines", True))
        self.set_property("tacking_angle", self.get_property("tacking_angle", 90))
        self.set_property("layline_style", self.get_property("layline_style", {
            "color": "rgba(255, 0, 0, 0.6)",
            "weight": 2,
            "dashArray": "5,5"
        }))
        
        # &e�#n-�
        self.set_property("strategy_points", self.get_property("strategy_points", []))
        self.set_property("optimal_route", self.get_property("optimal_route", []))
        self.set_property("risk_areas", self.get_property("risk_areas", []))
    
    def add_mark(self, lat: float, lng: float, mark_type: str = "rounding", 
               options: Dict[str, Any] = None) -> None:
        """
        ������
        
        Parameters
        ----------
        lat : float
            �
        lng : float
            L�
        mark_type : str, optional
            ������, by default "rounding"
        options : Dict[str, Any], optional
            ����׷��, by default None
        """
        if options is None:
            options = {}
        
        marks = self.get_property("marks", [])
        marks.append({
            "lat": lat,
            "lng": lng,
            "type": mark_type,
            **options
        })
        
        self.set_property("marks", marks)
    
    def set_start_line(self, pin: Dict[str, float], boat: Dict[str, float], 
                      options: Dict[str, Any] = None) -> None:
        """
        ������-�
        
        Parameters
        ----------
        pin : Dict[str, float]
            ��Mn {lat, lng}
        boat : Dict[str, float]
            ���Mn {lat, lng}
        options : Dict[str, Any], optional
            ��׷��, by default None
        """
        if options is None:
            options = {}
        
        start_line = {
            "pin": pin,
            "boat": boat,
            **options
        }
        
        self.set_property("start_line", start_line)
    
    def add_strategy_point(self, lat: float, lng: float, point_type: str, 
                         options: Dict[str, Any] = None) -> None:
        """
        &eݤ�Ȓ��
        
        Parameters
        ----------
        lat : float
            �
        lng : float
            L�
        point_type : str
            ݤ�ȿ�� (advantage, caution, information, etc.)
        options : Dict[str, Any], optional
            ݤ�Ȫ׷��, by default None
        """
        if options is None:
            options = {}
        
        points = self.get_property("strategy_points", [])
        points.append({
            "lat": lat,
            "lng": lng,
            "type": point_type,
            **options
        })
        
        self.set_property("strategy_points", points)
    
    def add_risk_area(self, polygon: List[Dict[str, float]], risk_type: str = "caution", 
                    options: Dict[str, Any] = None) -> None:
        """
        깯�ꢒ��
        
        Parameters
        ----------
        polygon : List[Dict[str, float]]
            ���n��� [{lat, lng}, ...]
        risk_type : str, optional
            깯���, by default "caution"
        options : Dict[str, Any], optional
            �ꢪ׷��, by default None
        """
        if options is None:
            options = {}
        
        areas = self.get_property("risk_areas", [])
        areas.append({
            "polygon": polygon,
            "type": risk_type,
            **options
        })
        
        self.set_property("risk_areas", areas)
    
    def set_optimal_route(self, points: List[Dict[str, float]], 
                        options: Dict[str, Any] = None) -> None:
        """
        i��Ȓ-�
        
        Parameters
        ----------
        points : List[Dict[str, float]]
            ���
nݤ���� [{lat, lng}, ...]
        options : Dict[str, Any], optional
            ��Ȫ׷��, by default None
        """
        if options is None:
            options = {}
        
        route = {
            "points": points,
            **options
        }
        
        self.set_property("optimal_route", route)
    
    def get_chart_libraries(self) -> List[str]:
        """
        �;kŁj����n�Ȓ֗
        
        Returns
        -------
        List[str]
            ����nURL��
        """
        libraries = super().get_chart_libraries()
        
        # ��n����
        additional_libraries = [
            "https://cdn.jsdelivr.net/npm/leaflet-geometryutil@0.9.3/src/leaflet.geometryutil.min.js"
        ]
        
        return libraries + additional_libraries
    
    def render(self, context: Dict[str, Any]) -> str:
        """
        � �HTMLk�����
        
        Parameters
        ----------
        context : Dict[str, Any]
            ������ƭ��
            
        Returns
        -------
        str
            �����U�_HTML
        """
        # a���ï
        if not self.evaluate_conditions(context):
            return ""
        
        # ������K�����֗
        data = None
        if self.data_source and self.data_source in context:
            data = context[self.data_source]
        
        # ��ï���LjD4g����� oh:gM��F��
        if not data:
            data = {"points": []}
        
        # CSS����n֗
        css_style = self.get_css_styles()
        width, height = self.get_chart_dimensions()
        
        # ���n-�
        center_auto = self.get_property("center_auto", True)
        center_lat = self.get_property("center_lat", 35.4498)
        center_lng = self.get_property("center_lng", 139.6649)
        zoom_level = self.get_property("zoom_level", 13)
        
        # ����#n-�
        marks = self.get_property("marks", [])
        course_shape = self.get_property("course_shape", "windward_leeward")
        start_line = self.get_property("start_line", {})
        finish_line = self.get_property("finish_line", {})
        
        # ���#n-�
        show_laylines = self.get_property("show_laylines", True)
        tacking_angle = self.get_property("tacking_angle", 90)
        layline_style = self.get_property("layline_style", {
            "color": "rgba(255, 0, 0, 0.6)",
            "weight": 2,
            "dashArray": "5,5"
        })
        
        # &e�#n-�
        strategy_points = self.get_property("strategy_points", [])
        optimal_route = self.get_property("optimal_route", [])
        risk_areas = self.get_property("risk_areas", [])
        
        # ���n���h-�
        map_type = self.get_property("map_type", "osm")
        show_track = self.get_property("show_track", True)
        track_color = self.get_property("track_color", "rgba(54, 162, 235, 0.8)")
        track_width = self.get_property("track_width", 3)
        
        # ��������-�
        point_icons = self.get_property("point_icons", {
            "mark": {"color": "red", "icon": "map-marker-alt"},
            "start": {"color": "green", "icon": "flag"},
            "finish": {"color": "blue", "icon": "flag-checkered"},
            "advantage": {"color": "green", "icon": "thumbs-up"},
            "caution": {"color": "orange", "icon": "exclamation-triangle"},
            "information": {"color": "blue", "icon": "info-circle"},
            "default": {"color": "gray", "icon": "map-marker-alt"}
        })
        
        # ����JSON�Wk	�
        data_json = json.dumps(data)
        
        # ����#-��JSON�Wk	�
        course_config = {
            "marks": marks,
            "course_shape": course_shape,
            "start_line": start_line,
            "finish_line": finish_line,
            "show_laylines": show_laylines,
            "tacking_angle": tacking_angle,
            "layline_style": layline_style,
            "strategy_points": strategy_points,
            "optimal_route": optimal_route,
            "risk_areas": risk_areas
        }
        
        course_config_json = json.dumps(course_config)
        
        # ���-��JSON�Wk	�
        map_config = {
            "map_type": map_type,
            "center_auto": center_auto,
            "center": [center_lat, center_lng],
            "zoom_level": zoom_level,
            "show_track": show_track,
            "track_color": track_color,
            "track_width": track_width,
            "point_icons": point_icons
        }
        
        map_config_json = json.dumps(map_config)
        
        # ��nCSS����
        additional_css = """
        <style>
            .course-mark-icon {
                display: flex;
                align-items: center;
                justify-content: center;
                width: 32px;
                height: 32px;
                border-radius: 50%;
                color: white;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.3);
            }
            
            .start-line {
                stroke: green;
                stroke-width: 3;
                stroke-opacity: 0.8;
            }
            
            .finish-line {
                stroke: blue;
                stroke-width: 3;
                stroke-opacity: 0.8;
            }
            
            .layline {
                stroke-dasharray: 5, 5;
                stroke-opacity: 0.6;
            }
            
            .optimal-route {
                stroke: rgba(0, 128, 0, 0.8);
                stroke-width: 3;
                stroke-opacity: 0.8;
            }
            
            .risk-area {
                fill-opacity: 0.3;
                stroke-opacity: 0.6;
            }
            
            .risk-area-caution {
                fill: rgba(255, 165, 0, 0.3);
                stroke: rgba(255, 165, 0, 0.8);
            }
            
            .risk-area-danger {
                fill: rgba(255, 0, 0, 0.3);
                stroke: rgba(255, 0, 0, 0.8);
            }
            
            .risk-area-information {
                fill: rgba(0, 0, 255, 0.2);
                stroke: rgba(0, 0, 255, 0.6);
            }
            
            .course-popup {
                min-width: 200px;
            }
            
            .course-popup h4 {
                margin: 0 0 8px 0;
                padding-bottom: 5px;
                border-bottom: 1px solid #eee;
            }
            
            .course-popup p {
                margin: 5px 0;
            }
        </style>
        """
        
        # ��ׁ n�����
        html_content = f'''
        <div id="{self.element_id}" class="report-map-container" style="{css_style}">
            <!-- Leaflet CSS -->
            <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css" />
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css" />
            {additional_css}
            
            <div id="{self.map_id}" style="width: {width}; height: {height};"></div>
            
            <script>
                (function() {{
                    // ������
                    var courseData = {data_json};
                    var courseConfig = {course_config_json};
                    var mapConfig = {map_config_json};
                    
                    // ���
                    window.addEventListener('load', function() {{
                        // ���n\
                        var map = L.map('{self.map_id}');
                        
                        // ������nx�
                        var tileLayer;
                        switch(mapConfig.map_type) {{
                            case 'satellite':
                                tileLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
                                    attribution: 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community'
                                }});
                                break;
                            case 'nautical':
                                tileLayer = L.tileLayer('https://tiles.openseamap.org/seamark/{{z}}/{{x}}/{{y}}.png', {{
                                    attribution: 'Map data: &copy; <a href="http://www.openseamap.org">OpenSeaMap</a> contributors'
                                }});
                                break;
                            default:  // 'osm'
                                tileLayer = L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                                    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                                }});
                        }}
                        
                        // ����������k��
                        tileLayer.addTo(map);
                        
                        // ��ïݤ��h�4������
                        var trackPoints = [];
                        var latKey = 'lat';
                        var lngKey = 'lng';
                        
                        // ���bk�Xf�
                        if (Array.isArray(courseData)) {{
                            // Mbn4��ïݤ��n��'	
                            if (courseData.length > 0 && typeof courseData[0] === 'object') {{
                                // ����y�
                                if ('latitude' in courseData[0] && 'longitude' in courseData[0]) {{
                                    latKey = 'latitude';
                                    lngKey = 'longitude';
                                }} else if ('lat' in courseData[0] && 'lon' in courseData[0]) {{
                                    lngKey = 'lon';
                                }}
                                
                                // ��ïݤ�Ȓ��
                                for (var i = 0; i < courseData.length; i++) {{
                                    var point = courseData[i];
                                    if (typeof point === 'object' && point[latKey] && point[lngKey]) {{
                                        trackPoints.push([point[latKey], point[lngKey]]);
                                    }}
                                }}
                            }}
                        }} else if (typeof courseData === 'object') {{
                            // �ָ���bn4
                            if ('track' in courseData && Array.isArray(courseData.track)) {{
                                for (var i = 0; i < courseData.track.length; i++) {{
                                    var point = courseData.track[i];
                                    if (typeof point === 'object' && point[latKey] && point[lngKey]) {{
                                        trackPoints.push([point[latKey], point[lngKey]]);
                                    }}
                                }}
                            }} else if ('points' in courseData && Array.isArray(courseData.points)) {{
                                for (var i = 0; i < courseData.points.length; i++) {{
                                    var point = courseData.points[i];
                                    if (typeof point === 'object' && point[latKey] && point[lngKey]) {{
                                        trackPoints.push([point[latKey], point[lngKey]]);
                                    }}
                                }}
                            }}
                        }}
                        
                        // ������ג\
                        var trackLayer = L.layerGroup();
                        var markLayer = L.layerGroup();
                        var lineLayer = L.layerGroup();
                        var laylinesLayer = L.layerGroup();
                        var strategyLayer = L.layerGroup();
                        var routeLayer = L.layerGroup();
                        var riskLayer = L.layerGroup();
                        
                        // ���6�n_�n�ָ���
                        var overlays = {{}};
                        
                        // ��ï��\h:-�L��n4	
                        if (mapConfig.show_track && trackPoints.length > 0) {{
                            var trackLine = L.polyline(trackPoints, {{
                                color: mapConfig.track_color,
                                weight: mapConfig.track_width,
                                opacity: 0.8,
                                lineJoin: 'round'
                            }}).addTo(trackLayer);
                            
                            overlays["GPS��ï"] = trackLayer;
                            trackLayer.addTo(map);
                        }}
                        
                        // ���֤ji	�\
                        if (courseConfig.marks && courseConfig.marks.length > 0) {{
                            courseConfig.marks.forEach(function(mark) {{
                                // �����ג֗
                                var iconConfig = mapConfig.point_icons[mark.type] || mapConfig.point_icons.mark || mapConfig.point_icons.default;
                                var iconColor = mark.color || iconConfig.color || 'red';
                                var iconName = mark.icon || iconConfig.icon || 'map-marker-alt';
                                
                                // �������\
                                var markIcon = L.divIcon({{
                                    html: '<div class="course-mark-icon" style="background-color: ' + iconColor + ';"><i class="fas fa-' + iconName + '"></i></div>',
                                    className: 'course-mark-icon-wrapper',
                                    iconSize: [32, 32],
                                    iconAnchor: [16, 16]
                                }});
                                
                                // ����n\
                                var marker = L.marker([mark.lat, mark.lng], {{
                                    icon: markIcon,
                                    title: mark.name || 'Mark'
                                }}).addTo(markLayer);
                                
                                // ��ע��n��
                                var popupContent = '<div class="course-popup">';
                                popupContent += '<h4>' + (mark.name || 'Mark') + '</h4>';
                                if (mark.description) popupContent += '<p>' + mark.description + '</p>';
                                popupContent += '<p><strong>���:</strong> ' + mark.type + '</p>';
                                if (mark.rounding_direction) popupContent += '<p><strong>�*�:</strong> ' + mark.rounding_direction + '</p>';
                                popupContent += '</div>';
                                
                                marker.bindPopup(popupContent);
                            }});
                            
                            overlays["������"] = markLayer;
                            markLayer.addTo(map);
                        }}
                        
                        // ������\
                        if (courseConfig.start_line && courseConfig.start_line.pin && courseConfig.start_line.boat) {{
                            var pinPos = [courseConfig.start_line.pin.lat, courseConfig.start_line.pin.lng];
                            var boatPos = [courseConfig.start_line.boat.lat, courseConfig.start_line.boat.lng];
                            
                            // ������
                            var startLine = L.polyline([pinPos, boatPos], {{
                                color: 'green',
                                weight: 3,
                                opacity: 0.8,
                                className: 'start-line'
                            }}).addTo(lineLayer);
                            
                            // ��������
                            var pinIcon = L.divIcon({{
                                html: '<div class="course-mark-icon" style="background-color: green;"><i class="fas fa-flag"></i></div>',
                                className: 'course-mark-icon-wrapper',
                                iconSize: [24, 24],
                                iconAnchor: [12, 12]
                            }});
                            
                            var pinMarker = L.marker(pinPos, {{
                                icon: pinIcon,
                                title: 'Start Line (Pin End)'
                            }}).addTo(lineLayer);
                            
                            // ���ƣ�������
                            var boatIcon = L.divIcon({{
                                html: '<div class="course-mark-icon" style="background-color: green;"><i class="fas fa-ship"></i></div>',
                                className: 'course-mark-icon-wrapper',
                                iconSize: [24, 24],
                                iconAnchor: [12, 12]
                            }});
                            
                            var boatMarker = L.marker(boatPos, {{
                                icon: boatIcon,
                                title: 'Start Line (Boat End)'
                            }}).addTo(lineLayer);
                            
                            // ��ע��
                            var lineLength = L.GeometryUtil.length(startLine);
                            
                            startLine.bindPopup('<div class="course-popup"><h4>������</h4>' +
                                              '<p><strong>wU:</strong> ' + lineLength.toFixed(1) + ' m</p>' +
                                              '</div>');
                        }}
                        
                        // գ�÷���\
                        if (courseConfig.finish_line && courseConfig.finish_line.pin && courseConfig.finish_line.boat) {{
                            var pinPos = [courseConfig.finish_line.pin.lat, courseConfig.finish_line.pin.lng];
                            var boatPos = [courseConfig.finish_line.boat.lat, courseConfig.finish_line.boat.lng];
                            
                            // գ�÷���
                            var finishLine = L.polyline([pinPos, boatPos], {{
                                color: 'blue',
                                weight: 3,
                                opacity: 0.8,
                                className: 'finish-line'
                            }}).addTo(lineLayer);
                            
                            // ��������
                            var pinIcon = L.divIcon({{
                                html: '<div class="course-mark-icon" style="background-color: blue;"><i class="fas fa-flag"></i></div>',
                                className: 'course-mark-icon-wrapper',
                                iconSize: [24, 24],
                                iconAnchor: [12, 12]
                            }});
                            
                            var pinMarker = L.marker(pinPos, {{
                                icon: pinIcon,
                                title: 'Finish Line (Pin End)'
                            }}).addTo(lineLayer);
                            
                            // ���ƣ�������
                            var boatIcon = L.divIcon({{
                                html: '<div class="course-mark-icon" style="background-color: blue;"><i class="fas fa-ship"></i></div>',
                                className: 'course-mark-icon-wrapper',
                                iconSize: [24, 24],
                                iconAnchor: [12, 12]
                            }});
                            
                            var boatMarker = L.marker(boatPos, {{
                                icon: boatIcon,
                                title: 'Finish Line (Boat End)'
                            }}).addTo(lineLayer);
                            
                            // ��ע��
                            var lineLength = L.GeometryUtil.length(finishLine);
                            
                            finishLine.bindPopup('<div class="course-popup"><h4>գ�÷���</h4>' +
                                               '<p><strong>wU:</strong> ' + lineLength.toFixed(1) + ' m</p>' +
                                               '</div>');
                        }}
                        
                        // ���������k��
                        overlays["����/գ�÷���"] = lineLayer;
                        lineLayer.addTo(map);
                        
                        // ���\�
���xn�����	
                        if (courseConfig.show_laylines && courseConfig.marks && courseConfig.marks.length > 0) {{
                            // �
�����Y
                            var windwardMark = null;
                            var leewardMark = null;
                            
                            // ���b�k�eDf�
������y�
                            if (courseConfig.course_shape === 'windward_leeward') {{
                                // �
�����n4h�n����(
                                if (courseConfig.marks.length >= 2) {{
                                    windwardMark = courseConfig.marks[0];
                                    leewardMark = courseConfig.marks[1];
                                }}
                            }} else {{
                                // ]��n4:�k�U�_�����Y
                                for (var i = 0; i < courseConfig.marks.length; i++) {{
                                    var mark = courseConfig.marks[i];
                                    if (mark.type === 'windward' || mark.role === 'windward') {{
                                        windwardMark = mark;
                                    }} else if (mark.type === 'leeward' || mark.role === 'leeward') {{
                                        leewardMark = mark;
                                    }}
                                }}
                            }}
                            
                            // �í�Ҧ�թ��o90�	
                            var tackingAngle = courseConfig.tacking_angle || 90;
                            var halfAngle = tackingAngle / 2;
                            
                            // ��󹿤�
                            var laylineStyle = {{
                                color: 'rgba(255, 0, 0, 0.6)',
                                weight: 2,
                                dashArray: '5,5',
                                className: 'layline'
                            }};
                            
                            // ������n�������
                            if (courseConfig.layline_style) {{
                                laylineStyle = Object.assign({{}}, laylineStyle, courseConfig.layline_style);
                            }}
                            
                            // �
���K����O
                            if (windwardMark) {{
                                var windwardPos = [windwardMark.lat, windwardMark.lng];
                                
                                // ���nwU1km{i	
                                var laylineLength = 0.01;  // 1km�L�XM	
                                
                                // �tn���
                                var leftAngle = 180 + halfAngle;
                                var leftEndLat = windwardMark.lat + Math.sin(leftAngle * Math.PI / 180) * laylineLength;
                                var leftEndLng = windwardMark.lng + Math.cos(leftAngle * Math.PI / 180) * laylineLength;
                                
                                var leftLayline = L.polyline([windwardPos, [leftEndLat, leftEndLng]], laylineStyle).addTo(laylinesLayer);
                                
                                // �tn���
                                var rightAngle = 180 - halfAngle;
                                var rightEndLat = windwardMark.lat + Math.sin(rightAngle * Math.PI / 180) * laylineLength;
                                var rightEndLng = windwardMark.lng + Math.cos(rightAngle * Math.PI / 180) * laylineLength;
                                
                                var rightLayline = L.polyline([windwardPos, [rightEndLat, rightEndLng]], laylineStyle).addTo(laylinesLayer);
                                
                                // ��ע��
                                leftLayline.bindPopup('<div class="course-popup"><h4>����</h4>' +
                                                    '<p><strong>Ҧ:</strong> ' + leftAngle + '�</p></div>');
                                
                                rightLayline.bindPopup('<div class="course-popup"><h4>����</h4>' +
                                                     '<p><strong>Ҧ:</strong> ' + rightAngle + '�</p></div>');
                            }}
                            
                            // ����K����O
                            if (leewardMark) {{
                                var leewardPos = [leewardMark.lat, leewardMark.lng];
                                
                                // ���nwU1km{i	
                                var laylineLength = 0.01;  // 1km�L�XM	
                                
                                // �tn���
                                var leftAngle = halfAngle;
                                var leftEndLat = leewardMark.lat + Math.sin(leftAngle * Math.PI / 180) * laylineLength;
                                var leftEndLng = leewardMark.lng + Math.cos(leftAngle * Math.PI / 180) * laylineLength;
                                
                                var leftLayline = L.polyline([leewardPos, [leftEndLat, leftEndLng]], laylineStyle).addTo(laylinesLayer);
                                
                                // �tn���
                                var rightAngle = 360 - halfAngle;
                                var rightEndLat = leewardMark.lat + Math.sin(rightAngle * Math.PI / 180) * laylineLength;
                                var rightEndLng = leewardMark.lng + Math.cos(rightAngle * Math.PI / 180) * laylineLength;
                                
                                var rightLayline = L.polyline([leewardPos, [rightEndLat, rightEndLng]], laylineStyle).addTo(laylinesLayer);
                                
                                // ��ע��
                                leftLayline.bindPopup('<div class="course-popup"><h4>����</h4>' +
                                                    '<p><strong>Ҧ:</strong> ' + leftAngle + '�</p></div>');
                                
                                rightLayline.bindPopup('<div class="course-popup"><h4>����</h4>' +
                                                     '<p><strong>Ҧ:</strong> ' + rightAngle + '�</p></div>');
                            }}
                            
                            overlays["���"] = laylinesLayer;
                            laylinesLayer.addTo(map);
                        }}
                        
                        // &eݤ�Ȓ��
                        if (courseConfig.strategy_points && courseConfig.strategy_points.length > 0) {{
                            courseConfig.strategy_points.forEach(function(point) {{
                                // ݤ�ȿ�ג֗
                                var iconConfig = mapConfig.point_icons[point.type] || mapConfig.point_icons.default;
                                var iconColor = point.color || iconConfig.color || 'blue';
                                var iconName = point.icon || iconConfig.icon || 'info-circle';
                                
                                // ݤ�Ȣ���\
                                var pointIcon = L.divIcon({{
                                    html: '<div class="course-mark-icon" style="background-color: ' + iconColor + ';"><i class="fas fa-' + iconName + '"></i></div>',
                                    className: 'course-mark-icon-wrapper',
                                    iconSize: [32, 32],
                                    iconAnchor: [16, 16]
                                }});
                                
                                // ����n\
                                var marker = L.marker([point.lat, point.lng], {{
                                    icon: pointIcon,
                                    title: point.name || point.description || 'Strategy Point'
                                }}).addTo(strategyLayer);
                                
                                // ��ע��n��
                                var pointType = point.type === 'advantage' ? '	)ݤ��' : 
                                              point.type === 'caution' ? '�ݤ��' : 
                                              point.type === 'information' ? '�1ݤ��' : 
                                              '&eݤ��';
                                
                                var popupContent = '<div class="course-popup">';
                                if (point.name) popupContent += '<h4>' + point.name + '</h4>';
                                popupContent += '<p><strong>���:</strong> ' + pointType + '</p>';
                                if (point.description) popupContent += '<p>' + point.description + '</p>';
                                popupContent += '</div>';
                                
                                marker.bindPopup(popupContent);
                            }});
                            
                            overlays["&eݤ��"] = strategyLayer;
                            strategyLayer.addTo(map);
                        }}
                        
                        // i��Ȓ��
                        if (courseConfig.optimal_route && courseConfig.optimal_route.points && courseConfig.optimal_route.points.length > 0) {{
                            var routePoints = [];
                            
                            courseConfig.optimal_route.points.forEach(function(point) {{
                                routePoints.push([point.lat, point.lng]);
                            }});
                            
                            // �����\
                            var routeLine = L.polyline(routePoints, {{
                                color: 'rgba(0, 128, 0, 0.8)',
                                weight: 3,
                                opacity: 0.8,
                                lineJoin: 'round',
                                className: 'optimal-route'
                            }}).addTo(routeLayer);
                            
                            // ���n�
                            var description = courseConfig.optimal_route.description || '�h���';
                            var reason = courseConfig.optimal_route.reason || '';
                            
                            // ��ע��n��
                            var popupContent = '<div class="course-popup">';
                            popupContent += '<h4>' + description + '</h4>';
                            if (reason) popupContent += '<p>' + reason + '</p>';
                            popupContent += '</div>';
                            
                            routeLine.bindPopup(popupContent);
                            
                            overlays["i���"] = routeLayer;
                            routeLayer.addTo(map);
                        }}
                        
                        // 깯�ꢒ��
                        if (courseConfig.risk_areas && courseConfig.risk_areas.length > 0) {{
                            courseConfig.risk_areas.forEach(function(area) {{
                                // ���n����
                                var polygonPoints = [];
                                
                                area.polygon.forEach(function(point) {{
                                    polygonPoints.push([point.lat, point.lng]);
                                }});
                                
                                // 깯���k�X_����
                                var areaStyle = {{
                                    color: 'rgba(255, 165, 0, 0.8)',
                                    weight: 1,
                                    fillColor: 'rgba(255, 165, 0, 0.3)',
                                    fillOpacity: 0.3,
                                    className: 'risk-area risk-area-caution'
                                }};
                                
                                if (area.type === 'danger') {{
                                    areaStyle.color = 'rgba(255, 0, 0, 0.8)';
                                    areaStyle.fillColor = 'rgba(255, 0, 0, 0.3)';
                                    areaStyle.className = 'risk-area risk-area-danger';
                                }} else if (area.type === 'information') {{
                                    areaStyle.color = 'rgba(0, 0, 255, 0.6)';
                                    areaStyle.fillColor = 'rgba(0, 0, 255, 0.2)';
                                    areaStyle.className = 'risk-area risk-area-information';
                                }}
                                
                                // ���n\
                                var polygon = L.polygon(polygonPoints, areaStyle).addTo(riskLayer);
                                
                                // ��n�
                                var areaType = area.type === 'danger' ? 'qz��' : 
                                             area.type === 'caution' ? '���' : 
                                             area.type === 'information' ? '�1��' : 
                                             '��';
                                
                                var description = area.description || '';
                                
                                // ��ע��n��
                                var popupContent = '<div class="course-popup">';
                                popupContent += '<h4>' + areaType + '</h4>';
                                if (description) popupContent += '<p>' + description + '</p>';
                                popupContent += '</div>';
                                
                                polygon.bindPopup(popupContent);
                            }});
                            
                            overlays["깯��"] = riskLayer;
                            riskLayer.addTo(map);
                        }}
                        
                        // �����������
                        L.control.layers(null, overlays).addTo(map);
                        
                        // h:��-�
                        var bounds;
                        
                        // ���LB�p����+����
                        if (courseConfig.marks && courseConfig.marks.length > 0) {{
                            var points = [];
                            
                            courseConfig.marks.forEach(function(mark) {{
                                points.push([mark.lat, mark.lng]);
                            }});
                            
                            // ����/գ�÷�����
                            if (courseConfig.start_line && courseConfig.start_line.pin && courseConfig.start_line.boat) {{
                                points.push([courseConfig.start_line.pin.lat, courseConfig.start_line.pin.lng]);
                                points.push([courseConfig.start_line.boat.lat, courseConfig.start_line.boat.lng]);
                            }}
                            
                            if (courseConfig.finish_line && courseConfig.finish_line.pin && courseConfig.finish_line.boat) {{
                                points.push([courseConfig.finish_line.pin.lat, courseConfig.finish_line.pin.lng]);
                                points.push([courseConfig.finish_line.boat.lat, courseConfig.finish_line.boat.lng]);
                            }}
                            
                            bounds = L.latLngBounds(points);
                        }}
                        // ���LjO��ïLB�p��ï�+����
                        else if (trackPoints.length > 0) {{
                            bounds = L.latLngBounds(trackPoints);
                        }}
                        
                        // �ՄkhSLh:U���Fk���
                        if (mapConfig.center_auto && bounds) {{
                            map.fitBounds(bounds, {{
                                padding: [50, 50]  // Y}���
                            }});
                        }} else {{
                            map.setView(mapConfig.center, mapConfig.zoom_level);
                        }}
                        
                        // ��תָ��Ȓ�����kl�
                        window['{self.map_id}_map'] = map;
                    }});
                }})();
            </script>
        </div>
        '''
        
        return html_content
