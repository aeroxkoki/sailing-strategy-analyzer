"""
sailing_data_processor.reporting.elements.map.base_map_element

マップ要素の基底クラスを提供するモジュールです。
このモジュールは、すべてのマップ要素に共通する機能を定義します。
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import json
import uuid

from sailing_data_processor.reporting.elements.base_element import BaseElement
from sailing_data_processor.reporting.templates.template_model import ElementType, ElementModel


class BaseMapElement(BaseElement):
    """
    マップ要素の基底クラス
    
    すべてのマップタイプに共通する機能を提供します。
    """
    
    def __init__(self, model: Optional[ElementModel] = None, **kwargs):
        """
        初期化
        
        Parameters
        ----------
        model : Optional[ElementModel], optional
            要素モデル, by default None
        **kwargs : dict
            モデルが提供されない場合に使用されるプロパティ
        """
        # デフォルトでマップ要素タイプを設定
        if model is None and 'element_type' not in kwargs:
            kwargs['element_type'] = ElementType.MAP
        
        super().__init__(model, **kwargs)
        
        # 一意なマップIDを生成
        self.map_id = f"map_{self.element_id}_{str(uuid.uuid4())[:8]}"
        
        # デフォルトのマップ設定
        self.set_property("center_auto", self.get_property("center_auto", True))
        self.set_property("center_lat", self.get_property("center_lat", 35.4498))  # 横浜の緯度
        self.set_property("center_lng", self.get_property("center_lng", 139.6649))  # 横浜の経度
        self.set_property("zoom_level", self.get_property("zoom_level", 12))
        self.set_property("base_layer", self.get_property("base_layer", "osm"))  # OpenStreetMap
        
        # コントロール設定
        self.set_property("show_layer_control", self.get_property("show_layer_control", True))
        self.set_property("show_scale_control", self.get_property("show_scale_control", True))
        self.set_property("show_fullscreen_control", self.get_property("show_fullscreen_control", True))
        self.set_property("show_measure_control", self.get_property("show_measure_control", False))
    
    @property
    def data_source(self) -> str:
        """
        データソース名を取得
        
        Returns
        -------
        str
            データソース名
        """
        return self.get_property("data_source", "")
    
    @data_source.setter
    def data_source(self, value: str) -> None:
        """
        データソース名を設定
        
        Parameters
        ----------
        value : str
            データソース名
        """
        self.set_property("data_source", value)
    
    @property
    def center_auto(self) -> bool:
        """
        自動中心設定を取得
        
        Returns
        -------
        bool
            自動中心設定
        """
        return self.get_property("center_auto", True)
    
    @center_auto.setter
    def center_auto(self, value: bool) -> None:
        """
        自動中心設定を設定
        
        Parameters
        ----------
        value : bool
            自動中心設定
        """
        self.set_property("center_auto", value)
    
    def set_center(self, lat: float, lng: float) -> None:
        """
        マップの中心位置を設定
        
        Parameters
        ----------
        lat : float
            緯度
        lng : float
            経度
        """
        self.set_property("center_auto", False)
        self.set_property("center_lat", lat)
        self.set_property("center_lng", lng)
    
    def set_zoom(self, level: int) -> None:
        """
        マップのズームレベルを設定
        
        Parameters
        ----------
        level : int
            ズームレベル (通常は0-18の範囲)
        """
        self.set_property("zoom_level", max(0, min(18, level)))
    
    def set_base_layer(self, layer_type: str) -> None:
        """
        ベースレイヤーの種類を設定
        
        Parameters
        ----------
        layer_type : str
            レイヤータイプ ("osm", "satellite", "nautical", "topo"など)
        """
        self.set_property("base_layer", layer_type)
    
    def get_map_dimensions(self) -> Tuple[str, str]:
        """
        マップの寸法を取得
        
        Returns
        -------
        Tuple[str, str]
            幅と高さ (width, height) のCSS値のタプル
        """
        width = self.get_style("width", "100%")
        height = self.get_style("height", "500px")
        return width, height
    
    def get_map_libraries(self) -> List[str]:
        """
        マップの描画に必要なライブラリのリストを取得
        
        Returns
        -------
        List[str]
            ライブラリのURLリスト
        """
        return [
            "https://unpkg.com/leaflet@1.9.4/dist/leaflet.js",
            "https://unpkg.com/leaflet.markercluster@1.5.3/dist/leaflet.markercluster.js",
            "https://unpkg.com/leaflet-fullscreen@1.0.2/dist/Leaflet.fullscreen.min.js",
            "https://cdn.jsdelivr.net/npm/leaflet-measure@3.1.0/dist/leaflet-measure.min.js"
        ]
    
    def get_map_styles(self) -> List[str]:
        """
        マップの表示に必要なスタイルシートのリストを取得
        
        Returns
        -------
        List[str]
            スタイルシートのURLリスト
        """
        return [
            "https://unpkg.com/leaflet@1.9.4/dist/leaflet.css",
            "https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.css",
            "https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.Default.css",
            "https://unpkg.com/leaflet-fullscreen@1.0.2/dist/leaflet.fullscreen.css",
            "https://cdn.jsdelivr.net/npm/leaflet-measure@3.1.0/dist/leaflet-measure.css"
        ]
    
    def get_default_controls(self) -> Dict[str, bool]:
        """
        デフォルトのコントロール設定を取得
        
        Returns
        -------
        Dict[str, bool]
            コントロール設定
        """
        return {
            "layer_control": self.get_property("show_layer_control", True),
            "scale_control": self.get_property("show_scale_control", True),
            "fullscreen_control": self.get_property("show_fullscreen_control", True),
            "measure_control": self.get_property("show_measure_control", False)
        }
    
    def get_map_initialization_code(self, map_var: str = "map") -> str:
        """
        マップ初期化用のJavaScriptコードを取得
        
        Parameters
        ----------
        map_var : str, optional
            マップ変数名, by default "map"
            
        Returns
        -------
        str
            初期化コード
        """
        center_auto = self.get_property("center_auto", True)
        center_lat = self.get_property("center_lat", 35.4498)
        center_lng = self.get_property("center_lng", 139.6649)
        zoom_level = self.get_property("zoom_level", 12)
        
        # マップのベースレイヤー
        base_layer = self.get_property("base_layer", "osm")
        
        # コントロール設定
        controls = self.get_default_controls()
        
        return f"""
            // マップの初期化
            var {map_var} = L.map('{self.map_id}', {{
                fullscreenControl: {str(controls['fullscreen_control']).lower()}
            }});
            
            // ベースレイヤーの定義
            var osmLayer = L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            }});
            
            var satelliteLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
                attribution: 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community'
            }});
            
            var nauticalLayer = L.tileLayer('https://tiles.openseamap.org/seamark/{{z}}/{{x}}/{{y}}.png', {{
                attribution: 'Map data: &copy; <a href="http://www.openseamap.org">OpenSeaMap</a> contributors'
            }});
            
            var topoLayer = L.tileLayer('https://{{s}}.tile.opentopomap.org/{{z}}/{{x}}/{{y}}.png', {{
                attribution: 'Map data: &copy; <a href="https://www.opentopomap.org">OpenTopoMap</a> contributors'
            }});
            
            // レイヤーコントロールの設定
            var baseLayers = {{
                "OpenStreetMap": osmLayer,
                "Satellite": satelliteLayer,
                "Nautical": nauticalLayer,
                "Topographic": topoLayer
            }};
            
            // デフォルトベースレイヤーの選択
            var defaultBaseLayer;
            switch('{base_layer}') {{
                case 'satellite':
                    defaultBaseLayer = satelliteLayer;
                    break;
                case 'nautical':
                    defaultBaseLayer = nauticalLayer;
                    break;
                case 'topo':
                    defaultBaseLayer = topoLayer;
                    break;
                default:  // 'osm'
                    defaultBaseLayer = osmLayer;
            }}
            
            // デフォルトレイヤーをマップに追加
            defaultBaseLayer.addTo({map_var});
            
            // スケールコントロールの追加
            if ({str(controls['scale_control']).lower()}) {{
                L.control.scale({{
                    imperial: false,
                    maxWidth: 200
                }}).addTo({map_var});
            }}
            
            // 計測ツールの追加
            if ({str(controls['measure_control']).lower()}) {{
                var measureControl = new L.Control.Measure({{
                    position: 'topleft',
                    primaryLengthUnit: 'meters',
                    secondaryLengthUnit: 'kilometers',
                    primaryAreaUnit: 'sqmeters',
                    secondaryAreaUnit: 'hectares',
                    activeColor: '#ff8800',
                    completedColor: '#ff4400'
                }});
                
                measureControl.addTo({map_var});
            }}
            
            // レイヤーコントロールの追加
            if ({str(controls['layer_control']).lower()}) {{
                var overlayLayers = {{}};
                L.control.layers(baseLayers, overlayLayers).addTo({map_var});
            }}
            
            // 初期ビューの設定
            {map_var}.setView([{center_lat}, {center_lng}], {zoom_level});
            
            // グローバル変数として保存（外部からアクセス用）
            window['{self.map_id}_map'] = {map_var};
        """
    
    def render(self, context: Dict[str, Any]) -> str:
        """
        要素をHTMLにレンダリング
        
        Parameters
        ----------
        context : Dict[str, Any]
            レンダリングコンテキスト
            
        Returns
        -------
        str
            レンダリングされたHTML
        """
        # 条件チェック
        if not self.evaluate_conditions(context):
            return ""
        
        # CSSスタイルの取得
        css_style = self.get_css_styles()
        width, height = self.get_map_dimensions()
        
        # 必要なライブラリの取得
        libraries = self.get_map_libraries()
        library_tags = "\n".join([f'<script src="{lib}"></script>' for lib in libraries])
        
        # 必要なスタイルシートの取得
        styles = self.get_map_styles()
        style_tags = "\n".join([f'<link rel="stylesheet" href="{style}" />' for style in styles])
        
        # 初期化コード
        init_code = self.get_map_initialization_code()
        
        # マップ要素のレンダリング
        html_content = f'''
        <div id="{self.element_id}" class="report-map-container" style="{css_style}">
            <!-- Leaflet スタイルシート -->
            {style_tags}
            
            <style>
                #{self.map_id} {{
                    width: {width};
                    height: {height};
                }}
            </style>
            
            <div id="{self.map_id}" class="report-map"></div>
            
            <!-- Leaflet ライブラリ -->
            {library_tags}
            
            <script>
                (function() {{
                    // マップ初期化
                    window.addEventListener('load', function() {{
                        {init_code}
                    }});
                }})();
            </script>
        </div>
        '''
        
        return html_content
