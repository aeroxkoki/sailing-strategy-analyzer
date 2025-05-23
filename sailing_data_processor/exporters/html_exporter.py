# -*- coding: utf-8 -*-
"""
sailing_data_processor.exporters.html_exporter

セッション結果をHTML形式でエクスポートするモジュール
"""

import os
import json
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import datetime
import string
import html

from sailing_data_processor.project.session_model import SessionModel, SessionResult
from sailing_data_processor.exporters.session_exporter import SessionExporter


class HTMLExporter(SessionExporter):
    """
    HTML形式でセッション結果をエクスポートするクラス
    """
    
    def __init__(self, template_manager=None, config: Optional[Dict[str, Any]] = None):
        """
        初期化
        
        Parameters
        ----------
        template_manager : Optional, default=None
            テンプレート管理クラスのインスタンス
        config : Optional[Dict[str, Any]], default=None
            エクスポーター設定
        """
        super().__init__(template_manager, config)
        
        # HTMLエクスポートのベーステンプレート
        self.base_template = """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>${title}</title>
    <style>
    ${styles}
    </style>
    ${head_content}
</head>
<body>
    <div class="container">
        ${header}
        ${content}
        ${footer}
    </div>
    
    <script>
    ${scripts}
    </script>
</body>
</html>
"""
    
    def export_session(self, session: SessionModel, output_path: str, 
                       template: str = "default", options: Dict[str, Any] = None) -> str:
        """
        単一セッションをHTMLでエクスポート
        
        Parameters
        ----------
        session : SessionModel
            エクスポートするセッション
        output_path : str
            出力先パス
        template : str, optional
            使用するテンプレート名, by default "default"
        options : Dict[str, Any], optional
            エクスポートオプション, by default None
            
        Returns
        -------
        str
            エクスポートされたファイルのパス
        """
        options = options or {}
        
        # テンプレートの取得
        template_data = {}
        if self.template_manager:
            try:
                template_data = self.template_manager.get_template(template, "html")
            except Exception as e:
                self.warnings.append(f"テンプレートの読み込みに失敗しました: {e}")
                # デフォルト設定で続行
        
        # 出力先ディレクトリの確認
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # HTMLコンテンツの生成
        title = html.escape(session.name)
        styles = self._generate_styles(template_data)
        head_content = self._generate_head_content(template_data)
        header = self._generate_header(session, template_data, options)
        content = self._generate_content(session, template_data, options)
        footer = self._generate_footer(session, template_data, options)
        scripts = self._generate_scripts(template_data, options)
        
        # テンプレートの置換
        template = string.Template(self.base_template)
        html_content = template.substitute(
            title=title,
            styles=styles,
            head_content=head_content,
            header=header,
            content=content,
            footer=footer,
            scripts=scripts
        )
        
        # HTMLファイルの保存
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html_content)
        except Exception as e:
            self.errors.append(f"HTMLファイルの保存に失敗しました: {e}")
            raise
        
        return output_path
    
    def export_multiple_sessions(self, sessions: List[SessionModel], output_dir: str,
                                template: str = "default", options: Dict[str, Any] = None) -> List[str]:
        """
        複数セッションをHTMLでエクスポート
        
        Parameters
        ----------
        sessions : List[SessionModel]
            エクスポートするセッションのリスト
        output_dir : str
            出力先ディレクトリ
        template : str, optional
            使用するテンプレート名, by default "default"
        options : Dict[str, Any], optional
            エクスポートオプション, by default None
            
        Returns
        -------
        List[str]
            エクスポートされたファイルのパスのリスト
        """
        options = options or {}
        export_files = []
        
        # 出力ディレクトリの確認
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # 各セッションをエクスポート
        for session in sessions:
            try:
                # ファイル名の生成
                filename = self.generate_export_filename(session, "html")
                output_path = os.path.join(output_dir, filename)
                
                # エクスポート実行
                exported_file = self.export_session(session, output_path, template, options)
                export_files.append(exported_file)
            except Exception as e:
                self.errors.append(f"セッション '{session.name}' のエクスポートに失敗しました: {e}")
        
        # インデックスページの作成（オプション）
        if "create_index" in options and options["create_index"] and export_files:
            try:
                index_path = os.path.join(output_dir, "index.html")
                self._create_index_page(sessions, export_files, index_path, template_data=None)
                export_files.append(index_path)
            except Exception as e:
                self.errors.append(f"インデックスページの作成に失敗しました: {e}")
        
        return export_files
    
    def _generate_styles(self, template_data: Dict[str, Any]) -> str:
        """
        スタイルシートを生成
        
        Parameters
        ----------
        template_data : Dict[str, Any]
            テンプレートデータ
            
        Returns
        -------
        str
            CSS文字列
        """
        # デフォルトのスタイル
        default_styles = """
            :root {
                --primary-color: #1565C0;
                --secondary-color: #0D47A1;
                --accent-color: #4CAF50;
                --text-color: #212121;
                --light-text: #757575;
                --background-color: #FFFFFF;
                --section-background: #F5F5F5;
                --border-color: #E0E0E0;
            }
            
            * {
                box-sizing: border-box;
                margin: 0;
                padding: 0;
            }
            
            body {
                font-family: 'Arial', 'Helvetica', sans-serif;
                line-height: 1.6;
                color: var(--text-color);
                background-color: var(--background-color);
            }
            
            .container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }
            
            header {
                margin-bottom: 30px;
                text-align: center;
                padding-bottom: 20px;
                border-bottom: 1px solid var(--border-color);
            }
            
            header h1 {
                color: var(--primary-color);
                margin-bottom: 10px;
            }
            
            header .session-date {
                color: var(--light-text);
                font-style: italic;
            }
            
            .section {
                margin-bottom: 40px;
                padding: 20px;
                background-color: var(--section-background);
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            
            .section h2 {
                color: var(--primary-color);
                margin-bottom: 15px;
                padding-bottom: 10px;
                border-bottom: 1px solid var(--border-color);
            }
            
            .metadata-item {
                margin-bottom: 10px;
            }
            
            .metadata-item .label {
                font-weight: bold;
                color: var(--secondary-color);
                margin-right: 10px;
            }
            
            .tags span {
                display: inline-block;
                background-color: var(--accent-color);
                color: white;
                padding: 2px 8px;
                margin-right: 5px;
                margin-bottom: 5px;
                border-radius: 3px;
                font-size: 0.9em;
            }
            
            .results-list {
                list-style-type: none;
            }
            
            .results-list li {
                padding: 10px;
                border-bottom: 1px solid var(--border-color);
            }
            
            .results-list li:last-child {
                border-bottom: none;
            }
            
            footer {
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid var(--border-color);
                text-align: center;
                color: var(--light-text);
                font-size: 0.9em;
            }
            
            @media print {
                body {
                    background-color: white;
                }
                
                .container {
                    width: 100%;
                    max-width: none;
                    padding: 0;
                }
                
                .section {
                    break-inside: avoid;
                    page-break-inside: avoid;
                    background-color: white;
                    box-shadow: none;
                    border: 1px solid #ccc;
                }
                
                a {
                    text-decoration: none;
                    color: black;
                }
            }
        """
        
        # テンプレートからスタイル設定を取得
        css_framework = template_data.get("styles", {}).get("css_framework", "")
        theme = template_data.get("styles", {}).get("theme", "light")
        custom_css = template_data.get("styles", {}).get("custom_css", "")
        
        # CSSフレームワークを含める場合
        framework_css = ""
        if css_framework.lower() == "bootstrap":
            framework_css = """
                <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
            """
        
        # テーマに応じたスタイル調整
        theme_css = ""
        if theme == "dark":
            theme_css = """
                :root {
                    --primary-color: #2196F3;
                    --secondary-color: #64B5F6;
                    --accent-color: #4CAF50;
                    --text-color: #EEEEEE;
                    --light-text: #BDBDBD;
                    --background-color: #121212;
                    --section-background: #1E1E1E;
                    --border-color: #333333;
                }
                
                body {
                    background-color: var(--background-color);
                    color: var(--text-color);
                }
                
                .section {
                    background-color: var(--section-background);
                }
            """
        
        # スタイルを統合
        all_styles = default_styles + theme_css + custom_css
        return all_styles
    
    def _generate_head_content(self, template_data: Dict[str, Any]) -> str:
        """
        HTMLヘッド要素内のコンテンツを生成
        
        Parameters
        ----------
        template_data : Dict[str, Any]
            テンプレートデータ
            
        Returns
        -------
        str
            HTMLヘッド要素内のコンテンツ
        """
        # テンプレートからメタデータ設定を取得
        include_interactive = template_data.get("metadata", {}).get("include_interactive", True)
        
        head_content = ""
        
        # インタラクティブ要素を含める場合は、必要なライブラリを追加
        if include_interactive:
            head_content += """
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/chart.js@3.7.1/dist/chart.min.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@3.7.1/dist/chart.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/leaflet@1.7.1/dist/leaflet.js"></script>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet@1.7.1/dist/leaflet.css">
            """
        
        return head_content
    
    def _generate_header(self, session: SessionModel, template_data: Dict[str, Any], options: Dict[str, Any]) -> str:
        """
        HTMLヘッダーセクションを生成
        
        Parameters
        ----------
        session : SessionModel
            セッション
        template_data : Dict[str, Any]
            テンプレートデータ
        options : Dict[str, Any]
            エクスポートオプション
            
        Returns
        -------
        str
            HTMLヘッダーセクション
        """
        # 日付文字列の生成
        date_str = ""
        if session.event_date:
            try:
                if isinstance(session.event_date, str):
                    event_date = datetime.datetime.fromisoformat(session.event_date)
                    date_str = event_date.strftime("%Y年%m月%d日")
                else:
                    date_str = session.event_date.strftime("%Y年%m月%d日")
            except (ValueError, AttributeError):
                pass
        
        location_str = f"<p class='session-location'>{html.escape(session.location)}</p>" if session.location else ""
        
        # ヘッダーHTML
        header_html = f"""
    <header>
        <h1>{html.escape(session.name)}</h1>
        <p class='session-date'>{date_str}</p>
        {location_str}
    </header>
        """
        
        return header_html
    
    def _generate_content(self, session: SessionModel, template_data: Dict[str, Any], options: Dict[str, Any]) -> str:
        """
        HTMLコンテンツセクションを生成
        
        Parameters
        ----------
        session : SessionModel
            セッション
        template_data : Dict[str, Any]
            テンプレートデータ
        options : Dict[str, Any]
            エクスポートオプション
            
        Returns
        -------
        str
            HTMLコンテンツセクション
        """
        content_html = ""
        
        # テンプレートからセクション設定を取得
        sections = template_data.get("sections", [])
        
        # セクション順序の取得
        section_order = {}
        for section in sections:
            name = section.get("name", "")
            order = section.get("order", 0)
            enabled = section.get("enabled", True)
            
            if name and enabled:
                section_order[name] = order
        
        # メタデータセクション
        if "include_metadata" not in options or options["include_metadata"]:
            if "metadata" in section_order:
                metadata_html = self._generate_metadata_section(session, template_data)
                content_html += metadata_html
        
        # 結果セクション
        if "include_results" not in options or options["include_results"]:
            if "wind_analysis" in section_order or "strategy_points" in section_order or "performance" in section_order:
                results_html = self._generate_results_section(session, template_data, options)
                content_html += results_html
        
        return content_html
    
    def _generate_footer(self, session: SessionModel, template_data: Dict[str, Any], options: Dict[str, Any]) -> str:
        """
        HTMLフッターセクションを生成
        
        Parameters
        ----------
        session : SessionModel
            セッション
        template_data : Dict[str, Any]
            テンプレートデータ
        options : Dict[str, Any]
            エクスポートオプション
            
        Returns
        -------
        str
            HTMLフッターセクション
        """
        # 現在の日時
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # フッターHTML
        footer_html = f"""
    <footer>
        <p>出力日時: {timestamp}</p>
        <p>セーリング戦略分析システム</p>
    </footer>
        """
        
        return footer_html
    
    def _generate_scripts(self, template_data: Dict[str, Any], options: Dict[str, Any]) -> str:
        """
        HTMLスクリプトセクションを生成
        
        Parameters
        ----------
        template_data : Dict[str, Any]
            テンプレートデータ
        options : Dict[str, Any]
            エクスポートオプション
            
        Returns
        -------
        str
            HTMLスクリプトセクション
        """
        # インタラクティブ要素を含める場合のスクリプト
        include_interactive = template_data.get("metadata", {}).get("include_interactive", True)
        
        if not include_interactive:
            return ""
        
        # 基本的なJavaScriptコード
        scripts_html = """
    // ページ読み込み時の処理
    document.addEventListener('DOMContentLoaded', function() {
        // 折りたたみ可能なセクション
        const collapsibleHeaders = document.querySelectorAll('.collapsible-header');
        collapsibleHeaders.forEach(header => {
            header.addEventListener('click', function() {
                this.classList.toggle('active');
                const content = this.nextElementSibling;
                if (content.style.maxHeight) {
                    content.style.maxHeight = null;
                } else {
                    content.style.maxHeight = content.scrollHeight + 'px';
                }
            });
        });
        
        // グラフの初期化
        initCharts();
        
        // マップの初期化
        initMaps();
    });
    
    // グラフの初期化関数
    function initCharts() {
        const chartContainers = document.querySelectorAll('.chart-container');
        chartContainers.forEach(container => {
            // グラフデータはHTMLのdata属性から取得
            const chartType = container.dataset.chartType;
            const chartData = JSON.parse(container.dataset.chartData);
            const chartOptions = JSON.parse(container.dataset.chartOptions || '{}');
            
            const canvas = container.querySelector('canvas');
            if (canvas && chartData) {
                new Chart(canvas, {
                    type: chartType,
                    data: chartData,
                    options: chartOptions
                });
            }
        });
    }
    
    // マップの初期化関数
    function initMaps() {
        const mapContainers = document.querySelectorAll('.map-container');
        mapContainers.forEach(container => {
            // マップデータはHTMLのdata属性から取得
            const mapData = JSON.parse(container.dataset.mapData || '{}');
            const center = mapData.center || [35.6895, 139.6917]; // デフォルト: 東京
            const zoom = mapData.zoom || 13;
            
            const mapDiv = container.querySelector('.map');
            if (mapDiv) {
                const map = L.map(mapDiv).setView(center, zoom);
                
                L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                }).addTo(map);
                
                // ポイントやルートの追加
                if (mapData.points) {
                    mapData.points.forEach(point => {
                        L.marker([point.lat, point.lon])
                            .addTo(map)
                            .bindPopup(point.label || '');
                    });
                }
                
                if (mapData.routes) {
                    mapData.routes.forEach(route => {
                        L.polyline(route.points, {
                            color: route.color || 'blue',
                            weight: route.weight || 3
                        }).addTo(map);
                    });
                }
            }
        });
    }
    
    // 印刷用関数
    function printReport() {
        window.print();
    }
        """
        
        return scripts_html
    
    def _generate_metadata_section(self, session: SessionModel, template_data: Dict[str, Any]) -> str:
        """
        メタデータセクションを生成
        
        Parameters
        ----------
        session : SessionModel
            セッション
        template_data : Dict[str, Any]
            テンプレートデータ
            
        Returns
        -------
        str
            メタデータセクションのHTML
        """
        # テンプレートからセクション設定を取得
        section_title = "セッション情報"
        for section in template_data.get("sections", []):
            if section["name"] == "metadata":
                section_title = section.get("title", section_title)
                break
        
        # タグのHTML生成
        tags_html = ""
        if session.tags:
            tags_html = "<div class='tags'>"
            for tag in session.tags:
                tags_html += f"<span>{html.escape(tag)}</span>"
            tags_html += "</div>"
        
        # 重要度の変換
        importance_map = {
            "low": "低",
            "normal": "普通",
            "high": "高",
            "critical": "最重要"
        }
        importance_text = importance_map.get(session.importance, session.importance)
        
        # メタデータセクションのHTML
        metadata_html = f"""
    <section class='section' id='metadata'>
        <h2>{html.escape(section_title)}</h2>
        
        <div class='metadata-item'>
            <span class='label'>説明:</span>
            <span>{html.escape(session.description or '')}</span>
        </div>
        
        <div class='metadata-item'>
            <span class='label'>目的:</span>
            <span>{html.escape(session.purpose or '')}</span>
        </div>
        
        <div class='metadata-item'>
            <span class='label'>ステータス:</span>
            <span>{html.escape(session.status or '')}</span>
        </div>
        
        <div class='metadata-item'>
            <span class='label'>カテゴリ:</span>
            <span>{html.escape(session.category or '')}</span>
        </div>
        
        <div class='metadata-item'>
            <span class='label'>評価:</span>
            <span>{'★' * session.rating + '☆' * (5 - session.rating)}</span>
        </div>
        
        <div class='metadata-item'>
            <span class='label'>重要度:</span>
            <span>{html.escape(importance_text)}</span>
        </div>
        
        {f'''
        <div class='metadata-item'>
            <span class='label'>完了率:</span>
            <span>{session.completion_percentage}%</span>
        </div>
        ''' if hasattr(session, 'completion_percentage') else ''}
        
        {f'''
        <div class='metadata-item'>
            <span class='label'>タグ:</span>
            {tags_html}
        </div>
        ''' if session.tags else ''}
        
        {f'''
        <div class='metadata-item'>
            <span class='label'>船種:</span>
            <span>{html.escape(session.metadata.get('boat_type', ''))}</span>
        </div>
        ''' if 'boat_type' in session.metadata else ''}
        
        {f'''
        <div class='metadata-item'>
            <span class='label'>コース種類:</span>
            <span>{html.escape(session.metadata.get('course_type', ''))}</span>
        </div>
        ''' if 'course_type' in session.metadata else ''}
        
        {f'''
        <div class='metadata-item'>
            <span class='label'>風の状態:</span>
            <span>{html.escape(session.metadata.get('wind_condition', ''))}</span>
        </div>
        ''' if 'wind_condition' in session.metadata else ''}
        
        {f'''
        <div class='metadata-item'>
            <span class='label'>平均風速:</span>
            <span>{session.metadata.get('avg_wind_speed', '')} ノット</span>
        </div>
        ''' if 'avg_wind_speed' in session.metadata else ''}
    </section>
        """
        
        return metadata_html
    
    def _generate_results_section(self, session: SessionModel, template_data: Dict[str, Any], options: Dict[str, Any]) -> str:
        """
        結果セクションを生成
        
        Parameters
        ----------
        session : SessionModel
            セッション
        template_data : Dict[str, Any]
            テンプレートデータ
        options : Dict[str, Any]
            エクスポートオプション
            
        Returns
        -------
        str
            結果セクションのHTML
        """
        # 結果がない場合は空文字を返す
        if not hasattr(session, "results") or not session.results:
            return ""
        
        # テンプレートからセクション設定を取得
        sections = template_data.get("sections", [])
        section_order = {}
        
        for section in sections:
            name = section.get("name", "")
            order = section.get("order", 0)
            enabled = section.get("enabled", True)
            
            if name and enabled:
                section_order[name] = order
        
        # 結果セクションのHTML
        results_html = ""
        
        # 風向風速分析セクション
        if "wind_analysis" in section_order:
            wind_html = self._generate_wind_analysis_section(session, template_data)
            if wind_html:
                results_html += wind_html
        
        # 戦略ポイントセクション
        if "strategy_points" in section_order:
            strategy_html = self._generate_strategy_points_section(session, template_data)
            if strategy_html:
                results_html += strategy_html
        
        # パフォーマンスセクション
        if "performance" in section_order:
            performance_html = self._generate_performance_section(session, template_data)
            if performance_html:
                results_html += performance_html
        
        # 結果がない場合はデフォルトセクションを表示
        if not results_html:
            results_html = f"""
    <section class='section' id='results'>
        <h2>分析結果</h2>
        
        <div class='results-summary'>
            <p>分析結果数: {len(session.results)}</p>
        </div>
        
        <ul class='results-list'>
            <!-- 結果リスト（実際のデータに基づいて生成） -->
            <li>結果の詳細表示には、セッション結果マネージャーからのデータが必要です。</li>
        </ul>
    </section>
            """
        
        return results_html
    
    def _generate_wind_analysis_section(self, session: SessionModel, template_data: Dict[str, Any]) -> str:
        """
        風向風速分析セクションを生成
        
        Parameters
        ----------
        session : SessionModel
            セッション
        template_data : Dict[str, Any]
            テンプレートデータ
            
        Returns
        -------
        str
            風向風速分析セクションのHTML
        """
        # 風分析の結果を取得
        wind_results = None
        for result in session.results:
            if result.result_type == "wind_analysis":
                wind_results = result
                break
        
        if not wind_results:
            return ""
        
        # セクションタイトルを取得
        section_title = "風向風速分析"
        for section in template_data.get("sections", []):
            if section.get("name") == "wind_analysis":
                section_title = section.get("title", section_title)
                interactive = section.get("interactive", True)
                break
        
        # 風向風速のサマリーデータを作成
        wind_data = wind_results.data if hasattr(wind_results, "data") else {}
        
        # 基本統計情報
        avg_wind_speed = wind_data.get("avg_wind_speed", "N/A")
        avg_wind_direction = wind_data.get("avg_wind_direction", "N/A")
        max_wind_speed = wind_data.get("max_wind_speed", "N/A")
        wind_shifts = wind_data.get("wind_shifts", [])
        wind_shift_count = len(wind_shifts) if isinstance(wind_shifts, list) else 0
        
        # グラフデータの準備
        has_time_series = False
        time_series_data = {}
        
        if "time_series" in wind_data and isinstance(wind_data["time_series"], dict):
            time_series = wind_data["time_series"]
            if "timestamps" in time_series and "wind_speed" in time_series and "wind_direction" in time_series:
                has_time_series = True
                time_series_data = {
                    "labels": [ts.split("T")[1].split("+")[0].split(".")[0] for ts in time_series["timestamps"]],
                    "wind_speed": time_series["wind_speed"],
                    "wind_direction": time_series["wind_direction"]
                }
        
        # 風向風速のグラフを生成
        wind_charts_html = ""
        if has_time_series and interactive:
            # 風速グラフデータ
            wind_speed_data = {
                "labels": time_series_data["labels"],
                "datasets": [{
                    "label": "風速 (ノット)",
                    "data": time_series_data["wind_speed"],
                    "borderColor": "#2196F3",
                    "backgroundColor": "rgba(33, 150, 243, 0.2)",
                    "borderWidth": 2,
                    "fill": True
                }]
            }
            
            # 風向グラフデータ
            wind_direction_data = {
                "labels": time_series_data["labels"],
                "datasets": [{
                    "label": "風向 (度)",
                    "data": time_series_data["wind_direction"],
                    "borderColor": "#FF9800",
                    "backgroundColor": "rgba(255, 152, 0, 0.2)",
                    "borderWidth": 2,
                    "fill": True
                }]
            }
            
            # グラフオプション
            chart_options = {
                "scales": {
                    "y": {
                        "beginAtZero": True
                    }
                },
                "responsive": True,
                "maintainAspectRatio": False
            }
            
            # チャートHTML
            wind_charts_html = f"""
        <div class="chart-container" style="position: relative; height:300px; margin-bottom: 20px;" 
            data-chart-type="line" 
            data-chart-data='{{"labels": {json.dumps(time_series_data["labels"])}, 
                            "datasets": [{{"label": "風速 (ノット)", 
                                          "data": {json.dumps(time_series_data["wind_speed"])}, 
                                          "borderColor": "#2196F3", 
                                          "backgroundColor": "rgba(33, 150, 243, 0.2)", 
                                          "borderWidth": 2, 
                                          "fill": true}}]}}' 
            data-chart-options='{{"scales": {{"y": {{"beginAtZero": true}}}}, "responsive": true, "maintainAspectRatio": false}}'>
            <canvas id="windSpeedChart"></canvas>
        </div>
        
        <div class="chart-container" style="position: relative; height:300px; margin-bottom: 20px;" 
            data-chart-type="line" 
            data-chart-data='{{"labels": {json.dumps(time_series_data["labels"])}, 
                            "datasets": [{{"label": "風向 (度)", 
                                          "data": {json.dumps(time_series_data["wind_direction"])}, 
                                          "borderColor": "#FF9800", 
                                          "backgroundColor": "rgba(255, 152, 0, 0.2)", 
                                          "borderWidth": 2, 
                                          "fill": true}}]}}' 
            data-chart-options='{{"scales": {{"y": {{"beginAtZero": true, "max": 360}}}}, "responsive": true, "maintainAspectRatio": false}}'>
            <canvas id="windDirectionChart"></canvas>
        </div>
            """
        
        # 風シフトの情報
        wind_shifts_html = ""
        if wind_shift_count > 0:
            shift_items = ""
            for i, shift in enumerate(wind_shifts):
                shift_time = shift.get("timestamp", "").split("T")[1].split("+")[0].split(".")[0] if "timestamp" in shift else "不明"
                shift_type = shift.get("type", "不明")
                shift_magnitude = shift.get("magnitude", 0)
                
                shift_items += f"""
                <li>
                    <span class="shift-time">{shift_time}</span>: 
                    <span class="shift-type">{shift_type}</span> 
                    <span class="shift-magnitude">({shift_magnitude}°)</span>
                </li>
                """
            
            wind_shifts_html = f"""
        <div class="wind-shifts">
            <h3>風向シフト情報 ({wind_shift_count}件)</h3>
            <ul class="shift-list">
                {shift_items}
            </ul>
        </div>
            """
        
        # 風向風速分析セクションのHTML
        wind_analysis_html = f"""
    <section class='section' id='wind-analysis'>
        <h2>{html.escape(section_title)}</h2>
        
        <div class="stats-summary">
            <div class="stat-item">
                <span class="stat-label">平均風速:</span>
                <span class="stat-value">{avg_wind_speed} ノット</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">平均風向:</span>
                <span class="stat-value">{avg_wind_direction}°</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">最大風速:</span>
                <span class="stat-value">{max_wind_speed} ノット</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">風向シフト回数:</span>
                <span class="stat-value">{wind_shift_count} 回</span>
            </div>
        </div>
        
        {wind_charts_html}
        
        {wind_shifts_html}
    </section>
        """
        
        return wind_analysis_html
    
    def _generate_strategy_points_section(self, session: SessionModel, template_data: Dict[str, Any]) -> str:
        """
        戦略ポイントセクションを生成
        
        Parameters
        ----------
        session : SessionModel
            セッション
        template_data : Dict[str, Any]
            テンプレートデータ
            
        Returns
        -------
        str
            戦略ポイントセクションのHTML
        """
        # 戦略ポイントの結果を取得
        strategy_results = None
        for result in session.results:
            if result.result_type == "strategy_points":
                strategy_results = result
                break
        
        if not strategy_results:
            return ""
        
        # セクションタイトルを取得
        section_title = "戦略ポイント分析"
        for section in template_data.get("sections", []):
            if section.get("name") == "strategy_points":
                section_title = section.get("title", section_title)
                interactive = section.get("interactive", True)
                break
        
        # 戦略ポイントデータを作成
        strategy_data = strategy_results.data if hasattr(strategy_results, "data") else {}
        
        # 戦略ポイントのリスト
        strategy_points = strategy_data.get("points", [])
        point_count = len(strategy_points) if isinstance(strategy_points, list) else 0
        
        # 戦略ポイントの種類別集計
        point_types = {}
        for point in strategy_points:
            point_type = point.get("type", "unknown")
            if point_type in point_types:
                point_types[point_type] += 1
            else:
                point_types[point_type] = 1
        
        # 戦略ポイントの表示
        points_list_html = ""
        if point_count > 0:
            points_items = ""
            for i, point in enumerate(strategy_points):
                point_time = point.get("timestamp", "").split("T")[1].split("+")[0].split(".")[0] if "timestamp" in point else "不明"
                point_type = point.get("type", "不明")
                point_score = point.get("score", 0)
                point_description = point.get("description", "")
                
                # スコアに基づいて色分け
                score_class = "low-score"
                if point_score >= 0.7:
                    score_class = "high-score"
                elif point_score >= 0.4:
                    score_class = "medium-score"
                
                points_items += f"""
                <li class="strategy-point {point_type.lower()}">
                    <div class="point-header">
                        <span class="point-time">{point_time}</span>
                        <span class="point-type">{point_type}</span>
                        <span class="point-score {score_class}">{point_score:.2f}</span>
                    </div>
                    <div class="point-description">{html.escape(point_description)}</div>
                </li>
                """
            
            points_list_html = f"""
        <div class="strategy-points-list">
            <h3>戦略ポイント ({point_count}件)</h3>
            <ul class="points-list">
                {points_items}
            </ul>
        </div>
            """
        
        # 戦略ポイントの種類別グラフ
        strategy_chart_html = ""
        if point_types and interactive:
            chart_data = {
                "labels": list(point_types.keys()),
                "datasets": [{
                    "label": "戦略ポイント数",
                    "data": list(point_types.values()),
                    "backgroundColor": [
                        "rgba(255, 99, 132, 0.5)",
                        "rgba(54, 162, 235, 0.5)",
                        "rgba(255, 206, 86, 0.5)",
                        "rgba(75, 192, 192, 0.5)",
                        "rgba(153, 102, 255, 0.5)"
                    ]
                }]
            }
            
            strategy_chart_html = f"""
        <div class="chart-container" style="position: relative; height:300px; margin-bottom: 20px;" 
            data-chart-type="bar" 
            data-chart-data='{{"labels": {json.dumps(list(point_types.keys()))}, 
                            "datasets": [{{"label": "戦略ポイント数", 
                                          "data": {json.dumps(list(point_types.values()))}, 
                                          "backgroundColor": [
                                              "rgba(255, 99, 132, 0.5)",
                                              "rgba(54, 162, 235, 0.5)",
                                              "rgba(255, 206, 86, 0.5)",
                                              "rgba(75, 192, 192, 0.5)",
                                              "rgba(153, 102, 255, 0.5)"
                                          ]}}]}}' 
            data-chart-options='{{"scales": {{"y": {{"beginAtZero": true}}}}, "responsive": true, "maintainAspectRatio": false}}'>
            <canvas id="strategyPointsChart"></canvas>
        </div>
            """
        
        # マップ表示
        map_html = ""
        if interactive and point_count > 0 and all("lat" in point and "lon" in point for point in strategy_points):
            # マップデータを作成
            map_points = []
            for point in strategy_points:
                map_points.append({
                    "lat": point.get("lat"),
                    "lon": point.get("lon"),
                    "label": f"{point.get('type', '')}: {point.get('score', 0):.2f}"
                })
            
            # 航跡データがある場合
            track_data = []
            if "track" in strategy_data and isinstance(strategy_data["track"], list):
                for track_point in strategy_data["track"]:
                    if "lat" in track_point and "lon" in track_point:
                        track_data.append([track_point["lat"], track_point["lon"]])
            
            map_routes = []
            if track_data:
                map_routes.append({
                    "points": track_data,
                    "color": "blue",
                    "weight": 3
                })
            
            # マップデータをJSONに変換
            map_data = {
                "center": map_points[0] if map_points else [35.6895, 139.6917],  # デフォルト: 東京
                "zoom": 15,
                "points": map_points,
                "routes": map_routes
            }
            
            map_html = f"""
        <div class="map-container" style="height: 400px; margin-bottom: 20px;" data-map-data='{json.dumps(map_data)}'>
            <div id="strategyMap" class="map" style="height: 100%;"></div>
        </div>
            """
        
        # 戦略ポイントセクションのHTML
        strategy_points_html = f"""
    <section class='section' id='strategy-points'>
        <h2>{html.escape(section_title)}</h2>
        
        <div class="stats-summary">
            <div class="stat-item">
                <span class="stat-label">戦略ポイント総数:</span>
                <span class="stat-value">{point_count}</span>
            </div>
            
            {' '.join([f'<div class="stat-item"><span class="stat-label">{k}:</span><span class="stat-value">{v}</span></div>' for k, v in point_types.items()])}
        </div>
        
        {strategy_chart_html}
        
        {map_html}
        
        {points_list_html}
    </section>
        """
        
        return strategy_points_html
    
    def _generate_performance_section(self, session: SessionModel, template_data: Dict[str, Any]) -> str:
        """
        パフォーマンスセクションを生成
        
        Parameters
        ----------
        session : SessionModel
            セッション
        template_data : Dict[str, Any]
            テンプレートデータ
            
        Returns
        -------
        str
            パフォーマンスセクションのHTML
        """
        # パフォーマンスの結果を取得
        performance_results = None
        for result in session.results:
            if result.result_type == "performance":
                performance_results = result
                break
        
        if not performance_results:
            return ""
        
        # セクションタイトルを取得
        section_title = "パフォーマンス分析"
        for section in template_data.get("sections", []):
            if section.get("name") == "performance":
                section_title = section.get("title", section_title)
                interactive = section.get("interactive", True)
                break
        
        # パフォーマンスデータを作成
        performance_data = performance_results.data if hasattr(performance_results, "data") else {}
        
        # 基本統計情報
        overall_score = performance_data.get("overall_score", 0)
        upwind_vmg = performance_data.get("upwind_vmg", 0)
        downwind_vmg = performance_data.get("downwind_vmg", 0)
        tack_efficiency = performance_data.get("tack_efficiency", 0)
        gybe_efficiency = performance_data.get("gybe_efficiency", 0)
        
        # 詳細スコア
        detailed_scores = performance_data.get("detailed_scores", {})
        
        # スコアグラフを生成
        score_chart_html = ""
        if detailed_scores and interactive:
            chart_labels = list(detailed_scores.keys())
            chart_data = list(detailed_scores.values())
            
            score_chart_html = f"""
        <div class="chart-container" style="position: relative; height:300px; margin-bottom: 20px;" 
            data-chart-type="radar" 
            data-chart-data='{{"labels": {json.dumps(chart_labels)}, 
                            "datasets": [{{"label": "パフォーマンススコア", 
                                          "data": {json.dumps(chart_data)}, 
                                          "backgroundColor": "rgba(54, 162, 235, 0.2)", 
                                          "borderColor": "rgb(54, 162, 235)", 
                                          "pointBackgroundColor": "rgb(54, 162, 235)", 
                                          "pointBorderColor": "#fff", 
                                          "pointHoverBackgroundColor": "#fff", 
                                          "pointHoverBorderColor": "rgb(54, 162, 235)"}}]}}' 
            data-chart-options='{{"scales": {{"r": {{"beginAtZero": true, "max": 1}}}}, "responsive": true, "maintainAspectRatio": false}}'>
            <canvas id="performanceRadarChart"></canvas>
        </div>
            """
        
        # 速度プロファイルグラフ
        speed_chart_html = ""
        if "speed_profile" in performance_data and interactive:
            speed_profile = performance_data["speed_profile"]
            
            if "timestamps" in speed_profile and "speed" in speed_profile:
                labels = [ts.split("T")[1].split("+")[0].split(".")[0] for ts in speed_profile["timestamps"]]
                speed_data = speed_profile["speed"]
                
                speed_chart_html = f"""
        <div class="chart-container" style="position: relative; height:300px; margin-bottom: 20px;" 
            data-chart-type="line" 
            data-chart-data='{{"labels": {json.dumps(labels)}, 
                            "datasets": [{{"label": "速度 (ノット)", 
                                          "data": {json.dumps(speed_data)}, 
                                          "borderColor": "#4CAF50", 
                                          "backgroundColor": "rgba(76, 175, 80, 0.2)", 
                                          "borderWidth": 2, 
                                          "fill": true}}]}}' 
            data-chart-options='{{"scales": {{"y": {{"beginAtZero": true}}}}, "responsive": true, "maintainAspectRatio": false}}'>
            <canvas id="speedProfileChart"></canvas>
        </div>
                """
        
        # 改善提案
        improvement_html = ""
        if "improvements" in performance_data and isinstance(performance_data["improvements"], list):
            improvements = performance_data["improvements"]
            
            if improvements:
                improvement_items = ""
                for i, imp in enumerate(improvements):
                    area = imp.get("area", "")
                    suggestion = imp.get("suggestion", "")
                    priority = imp.get("priority", "medium")
                    
                    priority_class = f"priority-{priority.lower()}"
                    
                    improvement_items += f"""
                    <li class="improvement-item {priority_class}">
                        <div class="improvement-header">
                            <span class="improvement-area">{html.escape(area)}</span>
                            <span class="improvement-priority">{priority}</span>
                        </div>
                        <div class="improvement-suggestion">{html.escape(suggestion)}</div>
                    </li>
                    """
                
                improvement_html = f"""
        <div class="improvements">
            <h3>改善提案</h3>
            <ul class="improvement-list">
                {improvement_items}
            </ul>
        </div>
                """
        
        # パフォーマンスセクションのHTML
        performance_html = f"""
    <section class='section' id='performance'>
        <h2>{html.escape(section_title)}</h2>
        
        <div class="overall-score">
            <div class="score-display">
                <div class="score-value">{overall_score:.2f}</div>
                <div class="score-label">総合スコア</div>
            </div>
        </div>
        
        <div class="stats-summary">
            <div class="stat-item">
                <span class="stat-label">上り（風上）VMG効率:</span>
                <span class="stat-value">{upwind_vmg:.2f}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">下り（風下）VMG効率:</span>
                <span class="stat-value">{downwind_vmg:.2f}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">タック効率:</span>
                <span class="stat-value">{tack_efficiency:.2f}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">ジャイブ効率:</span>
                <span class="stat-value">{gybe_efficiency:.2f}</span>
            </div>
        </div>
        
        {score_chart_html}
        
        {speed_chart_html}
        
        {improvement_html}
    </section>
        """
        
        return performance_html
        
    def _create_index_page(self, sessions: List[SessionModel], exported_files: List[str], 
                          output_path: str, template_data: Dict[str, Any] = None) -> str:
        """
        セッション一覧のインデックスページを作成
        
        Parameters
        ----------
        sessions : List[SessionModel]
            セッションのリスト
        exported_files : List[str]
            エクスポートされたファイルのパスのリスト
        output_path : str
            出力先パス
        template_data : Dict[str, Any], optional
            テンプレートデータ, by default None
            
        Returns
        -------
        str
            作成されたインデックスページのパス
        """
        # インデックスページのHTMLを生成
        title = "セッション一覧"
        
        # スタイルを生成
        styles = """
            :root {
                --primary-color: #1565C0;
                --secondary-color: #0D47A1;
                --accent-color: #4CAF50;
                --text-color: #212121;
                --light-text: #757575;
                --background-color: #FFFFFF;
                --section-background: #F5F5F5;
                --border-color: #E0E0E0;
            }
            
            * {
                box-sizing: border-box;
                margin: 0;
                padding: 0;
            }
            
            body {
                font-family: 'Arial', 'Helvetica', sans-serif;
                line-height: 1.6;
                color: var(--text-color);
                background-color: var(--background-color);
            }
            
            .container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }
            
            header {
                margin-bottom: 30px;
                text-align: center;
                padding-bottom: 20px;
                border-bottom: 1px solid var(--border-color);
            }
            
            header h1 {
                color: var(--primary-color);
                margin-bottom: 10px;
            }
            
            .session-list {
                list-style-type: none;
            }
            
            .session-item {
                padding: 15px;
                margin-bottom: 15px;
                background-color: var(--section-background);
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                transition: transform 0.2s ease-in-out;
            }
            
            .session-item:hover {
                transform: translateY(-3px);
                box-shadow: 0 4px 8px rgba(0,0,0,0.15);
            }
            
            .session-item a {
                text-decoration: none;
                color: var(--primary-color);
                font-weight: bold;
                font-size: 1.2em;
            }
            
            .session-item p {
                margin-top: 5px;
                color: var(--light-text);
            }
            
            .session-meta {
                margin-top: 10px;
                font-size: 0.9em;
            }
            
            .session-meta span {
                margin-right: 15px;
            }
            
            .tags span {
                display: inline-block;
                background-color: var(--accent-color);
                color: white;
                padding: 2px 8px;
                margin-right: 5px;
                margin-bottom: 5px;
                border-radius: 3px;
                font-size: 0.8em;
            }
            
            footer {
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid var(--border-color);
                text-align: center;
                color: var(--light-text);
                font-size: 0.9em;
            }
        """
        
        # セッション一覧の項目を生成
        session_items = ""
        for i, (session, file_path) in enumerate(zip(sessions, exported_files)):
            # ファイル名のみを取得
            file_name = os.path.basename(file_path)
            
            # 日付表示を作成
            date_str = ""
            if session.event_date:
                try:
                    if isinstance(session.event_date, str):
                        event_date = datetime.datetime.fromisoformat(session.event_date)
                        date_str = event_date.strftime("%Y年%m月%d日")
                    else:
                        date_str = session.event_date.strftime("%Y年%m月%d日")
                except (ValueError, AttributeError):
                    pass
            
            # タグのHTML
            tags_html = ""
            if session.tags:
                tags_html = "<div class='tags'>"
                for tag in session.tags:
                    tags_html += f"<span>{html.escape(tag)}</span>"
                tags_html += "</div>"
            
            # セッション項目のHTML
            session_items += f"""
            <li class='session-item'>
                <a href='{html.escape(file_name)}'>{html.escape(session.name)}</a>
                <p>{html.escape(session.description or '')}</p>
                <div class='session-meta'>
                    {f"<span>日付: {date_str}</span>" if date_str else ""}
                    <span>カテゴリ: {html.escape(session.category or '')}</span>
                    <span>評価: {'★' * session.rating + '☆' * (5 - session.rating)}</span>
                </div>
                {tags_html}
            </li>
            """
        
        # 現在の日時
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # インデックスHTMLの生成
        index_html = f"""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
    {styles}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{title}</h1>
            <p>エクスポートされたセッション: {len(sessions)}件</p>
        </header>
        
        <ul class="session-list">
            {session_items}
        </ul>
        
        <footer>
            <p>出力日時: {timestamp}</p>
            <p>セーリング戦略分析システム</p>
        </footer>
    </div>
</body>
</html>
        """
        
        # HTMLファイルの保存
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(index_html)
        except Exception as e:
            self.errors.append(f"インデックスページの保存に失敗しました: {e}")
            raise
        
        return output_path