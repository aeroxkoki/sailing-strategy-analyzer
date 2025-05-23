# -*- coding: utf-8 -*-
"""
PDFジェネレーターモジュール

PDFドキュメント生成機能を提供します。PDF ExporterとHTML Exporterで共有されます。
"""

import os
import logging
from pathlib import Path
import datetime
import io
from typing import Dict, List, Any, Optional, Union

# 追加のライブラリ
try:
    import reportlab
    from reportlab.lib.pagesizes import A4, letter, landscape
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
    from reportlab.lib import colors
    from reportlab.lib.units import inch, cm
    from reportlab.pdfgen import canvas
    from reportlab.platypus.flowables import KeepTogether
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

try:
    import weasyprint
    WEASYPRINT_AVAILABLE = True
except ImportError:
    WEASYPRINT_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

logger = logging.getLogger(__name__)

class PDFGenerator:
    """
    PDFドキュメント生成クラス
    
    ReportLabとWeasyPrintを使用したPDF生成機能を提供します。
    """
    
    def __init__(self, options: Dict[str, Any] = None):
        """
        初期化
        
        Parameters
        ----------
        options : Dict[str, Any], optional
            PDF生成オプション, by default None
        """
        self.options = options or {}
        self.errors = []
        self.warnings = []
    
    def add_error(self, message: str) -> None:
        """エラーメッセージを追加"""
        self.errors.append(message)
        logger.error(message)
    
    def add_warning(self, message: str) -> None:
        """警告メッセージを追加"""
        self.warnings.append(message)
        logger.warning(message)
    
    def generate_reportlab_pdf(self, data, template=None, **kwargs) -> Optional[bytes]:
        """
        ReportLabを使用してPDFを生成
        
        Parameters
        ----------
        data : Any
            エクスポートするデータ
        template : Optional[str]
            テンプレート名
        **kwargs : dict
            追加のパラメータ
            
        Returns
        -------
        Optional[bytes]
            PDF内容、失敗した場合はNone
        """
        if not REPORTLAB_AVAILABLE:
            self.add_error("ReportLabがインストールされていません。pip install reportlab を実行してください。")
            return None
        
        # ページサイズの設定
        page_size_name = self.options.get("page_size", "A4")
        if page_size_name == "A4":
            page_size = A4
        elif page_size_name == "letter":
            page_size = letter
        else:
            page_size = A4  # デフォルト
        
        # 向き（orientation）の設定
        orientation = self.options.get("orientation", "portrait")
        if orientation == "landscape":
            page_size = landscape(page_size)
        
        # マージンの設定
        margin = self.options.get("margin", {"top": 2.5, "right": 2.5, "bottom": 2.5, "left": 2.5})
        top_margin = margin.get("top", 2.5) * cm
        right_margin = margin.get("right", 2.5) * cm
        bottom_margin = margin.get("bottom", 2.5) * cm
        left_margin = margin.get("left", 2.5) * cm
        
        # PDFバッファの作成
        buffer = io.BytesIO()
        
        # PDF文書の作成
        doc = SimpleDocTemplate(
            buffer,
            pagesize=page_size,
            topMargin=top_margin,
            rightMargin=right_margin,
            bottomMargin=bottom_margin,
            leftMargin=left_margin
        )
        
        # スタイルの設定
        styles = getSampleStyleSheet()
        title_style = styles["Title"]
        heading1_style = styles["Heading1"]
        heading2_style = styles["Heading2"]
        normal_style = styles["Normal"]
        
        # フォントの設定
        font_name = self.options.get("font", "Helvetica")
        font_size = self.options.get("font_size", 11)
        
        # 文書タイトルの設定
        title = self.options.get("title", "セーリングデータ分析レポート")
        
        # 文書のコンテンツを構築
        content = []
        
        # タイトル
        content.append(Paragraph(title, title_style))
        content.append(Spacer(1, 12))
        
        # 生成日時
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        content.append(Paragraph(f"生成日時: {now}", normal_style))
        content.append(Spacer(1, 12))
        
        # データの種類に基づいて適切な処理
        if hasattr(data, 'metadata') and isinstance(data.metadata, dict):
            # メタデータセクション
            content.append(Paragraph("メタデータ", heading1_style))
            content.append(Spacer(1, 12))
            
            # メタデータテーブルの作成
            metadata_data = []
            metadata_data.append(["項目", "値"])
            
            for key, value in data.metadata.items():
                metadata_data.append([key, str(value)])
            
            # テーブルスタイルの設定
            table_style = TableStyle([
                ('BACKGROUND', (0, 0), (1, 0), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ])
            
            metadata_table = Table(metadata_data, colWidths=[doc.width*0.3, doc.width*0.7])
            metadata_table.setStyle(table_style)
            content.append(metadata_table)
            content.append(Spacer(1, 12))
        
        # データフレームを持つ場合
        if hasattr(data, 'data') and hasattr(data.data, 'head') and PANDAS_AVAILABLE:
            # データサマリーセクション
            content.append(Paragraph("データサマリー", heading1_style))
            content.append(Spacer(1, 12))
            
            # 基本統計量
            df = data.data
            stats_data = []
            stats_data.append(["項目", "値"])
            
            # データポイント数
            stats_data.append(["データポイント数", str(len(df))])
            
            # 時間範囲（あれば）
            if 'timestamp' in df.columns:
                try:
                    start_time = pd.to_datetime(df['timestamp'].min())
                    end_time = pd.to_datetime(df['timestamp'].max())
                    duration = end_time - start_time
                    
                    stats_data.append(["開始時刻", start_time.strftime('%Y-%m-%d %H:%M:%S')])
                    stats_data.append(["終了時刻", end_time.strftime('%Y-%m-%d %H:%M:%S')])
                    stats_data.append(["所要時間", str(duration)])
                except:
                    pass
            
            # 位置範囲（あれば）
            if 'latitude' in df.columns and 'longitude' in df.columns:
                stats_data.append(["緯度範囲", f"{df['latitude'].min():.6f} - {df['latitude'].max():.6f}"])
                stats_data.append(["経度範囲", f"{df['longitude'].min():.6f} - {df['longitude'].max():.6f}"])
            
            # 速度統計（あれば）
            if 'speed' in df.columns:
                stats_data.append(["平均速度", f"{df['speed'].mean():.2f}"])
                stats_data.append(["最大速度", f"{df['speed'].max():.2f}"])
            
            # 統計テーブルの作成
            stats_table = Table(stats_data, colWidths=[doc.width*0.3, doc.width*0.7])
            stats_table.setStyle(table_style)
            content.append(stats_table)
            content.append(Spacer(1, 12))
            
            # データテーブルセクション
            content.append(Paragraph("データテーブル", heading1_style))
            content.append(Spacer(1, 12))
            content.append(Paragraph("以下は記録されたデータの最初の10行です。", normal_style))
            content.append(Spacer(1, 12))
            
            # データテーブルの作成
            table_data = []
            
            # ヘッダー行
            headers = df.columns.tolist()
            table_data.append(headers)
            
            # データ行（最初の10行）
            for _, row in df.head(10).iterrows():
                row_data = []
                for col in headers:
                    value = row[col]
                    # 日時型の場合はフォーマット
                    if pd.api.types.is_datetime64_any_dtype(value):
                        value = pd.to_datetime(value).strftime('%Y-%m-%d %H:%M:%S')
                    row_data.append(str(value))
                table_data.append(row_data)
            
            # データテーブルスタイルの設定
            data_table_style = TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 1), (-1, -1), 8),  # データ行のフォントサイズ調整
            ])
            
            # テーブル列幅の計算（簡易的）
            col_widths = [doc.width / len(headers) for _ in headers]
            
            # データテーブルの作成
            data_table = Table(table_data, colWidths=col_widths)
            data_table.setStyle(data_table_style)
            content.append(data_table)
        
        # PDF文書の構築と出力
        doc.build(content)
        
        # バッファからPDFデータを取得
        pdf_data = buffer.getvalue()
        buffer.close()
        
        return pdf_data
    
    def generate_weasyprint_pdf(self, data, template=None, **kwargs) -> Optional[bytes]:
        """
        WeasyPrintを使用してHTMLからPDFを生成
        
        Parameters
        ----------
        data : Any
            エクスポートするデータ
        template : Optional[str]
            テンプレート名またはHTML文字列
        **kwargs : dict
            追加のパラメータ
            
        Returns
        -------
        Optional[bytes]
            PDF内容、失敗した場合はNone
        """
        if not WEASYPRINT_AVAILABLE:
            self.add_error("WeasyPrintがインストールされていません。pip install weasyprint を実行してください。")
            return None
        
        # HTMLを生成（HTMLテンプレートまたはデフォルトテンプレート）
        html_content = self._generate_html_content(data, template, **kwargs)
        
        # WeasyPrintを使用してPDFに変換
        pdf_document = weasyprint.HTML(string=html_content)
        
        # ページサイズの設定
        page_size_name = self.options.get("page_size", "A4")
        orientation = self.options.get("orientation", "portrait")
        
        # マージンの設定
        margin = self.options.get("margin", {"top": 2.5, "right": 2.5, "bottom": 2.5, "left": 2.5})
        
        # WeasyPrintのスタイルシートを生成
        css_content = f"""
        @page {{
            size: {page_size_name} {'landscape' if orientation == 'landscape' else 'portrait'};
            margin: {margin['top']}cm {margin['right']}cm {margin['bottom']}cm {margin['left']}cm;
        }}
        """
        
        # PDFを生成
        return pdf_document.write_pdf(stylesheets=[weasyprint.CSS(string=css_content)])
    
    def _generate_html_content(self, data, template=None, **kwargs) -> str:
        """
        HTML内容を生成
        
        Parameters
        ----------
        data : Any
            エクスポートするデータ
        template : Optional[str]
            テンプレート名またはHTML文字列
        **kwargs : dict
            追加のパラメータ
            
        Returns
        -------
        str
            HTML内容
        """
        # テンプレートが提供されていれば使用
        if template:
            # テンプレートがパスの場合、ファイルから読み込む
            if os.path.exists(template):
                with open(template, 'r', encoding='utf-8') as f:
                    html_template = f.read()
            else:
                # 直接テンプレート文字列として扱う
                html_template = template
        else:
            # デフォルトのHTMLテンプレート
            html_template = self._get_default_html_template()
        
        # データをHTMLにレンダリング
        try:
            # データとメタデータの取得
            metadata = {}
            if hasattr(data, 'metadata'):
                metadata = data.metadata
            
            # レポートタイトル
            title = self.options.get("title", "セーリングデータ分析レポート")
            if 'name' in metadata:
                title = f"{title} - {metadata['name']}"
            
            # HTML内のプレースホルダをデータで置換
            html_content = html_template
            
            # タイトルの置換
            html_content = html_content.replace("{title}", title)
            
            # 生成日時の置換
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            html_content = html_content.replace("{generated_date}", now)
            
            # メタデータの処理
            if "{metadata_table}" in html_content:
                metadata_html = "<table><tr><th>項目</th><th>値</th></tr>"
                for key, value in metadata.items():
                    metadata_html += f"<tr><td>{key}</td><td>{value}</td></tr>"
                metadata_html += "</table>"
                html_content = html_content.replace("{metadata_table}", metadata_html)
            
            # データテーブルの処理
            if "{data_table}" in html_content and hasattr(data, 'data'):
                df = data.data
                
                # ヘッダー行
                data_html = "<table><tr>"
                headers = df.columns.tolist()
                for header in headers:
                    data_html += f"<th>{header}</th>"
                data_html += "</tr>"
                
                # データ行（最初の100行まで）
                for _, row in df.head(100).iterrows():
                    data_html += "<tr>"
                    for col in headers:
                        value = row[col]
                        # 日時型の場合はフォーマット
                        if hasattr(value, 'strftime'):
                            value = value.strftime('%Y-%m-%d %H:%M:%S')
                        data_html += f"<td>{value}</td>"
                    data_html += "</tr>"
                
                data_html += "</table>"
                html_content = html_content.replace("{data_table}", data_html)
            
            # 基本統計量の処理
            if "{stats_table}" in html_content and hasattr(data, 'data'):
                df = data.data
                stats_html = "<table><tr><th>項目</th><th>値</th></tr>"
                
                # データポイント数
                stats_html += f"<tr><td>データポイント数</td><td>{len(df)}</td></tr>"
                
                # 時間範囲（あれば）
                if 'timestamp' in df.columns and PANDAS_AVAILABLE:
                    try:
                        start_time = pd.to_datetime(df['timestamp'].min())
                        end_time = pd.to_datetime(df['timestamp'].max())
                        duration = end_time - start_time
                        
                        stats_html += f"<tr><td>開始時刻</td><td>{start_time.strftime('%Y-%m-%d %H:%M:%S')}</td></tr>"
                        stats_html += f"<tr><td>終了時刻</td><td>{end_time.strftime('%Y-%m-%d %H:%M:%S')}</td></tr>"
                        stats_html += f"<tr><td>所要時間</td><td>{duration}</td></tr>"
                    except:
                        pass
                
                # 位置範囲（あれば）
                if 'latitude' in df.columns and 'longitude' in df.columns:
                    stats_html += f"<tr><td>緯度範囲</td><td>{df['latitude'].min():.6f} - {df['latitude'].max():.6f}</td></tr>"
                    stats_html += f"<tr><td>経度範囲</td><td>{df['longitude'].min():.6f} - {df['longitude'].max():.6f}</td></tr>"
                
                # 速度統計（あれば）
                if 'speed' in df.columns:
                    stats_html += f"<tr><td>平均速度</td><td>{df['speed'].mean():.2f}</td></tr>"
                    stats_html += f"<tr><td>最大速度</td><td>{df['speed'].max():.2f}</td></tr>"
                
                stats_html += "</table>"
                html_content = html_content.replace("{stats_table}", stats_html)
            
            return html_content
            
        except Exception as e:
            self.add_error(f"HTML生成中にエラーが発生しました: {str(e)}")
            logger.error(f"HTML generation failed: {str(e)}", exc_info=True)
            return self._get_default_html_template()
    
    def _get_default_html_template(self) -> str:
        """
        デフォルトのHTMLテンプレートを取得
        
        Returns
        -------
        str
            HTMLテンプレート
        """
        return """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body { font-family: Helvetica, Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        h1, h2, h3 { color: #2c3e50; }
        h1 { border-bottom: 2px solid #3498db; padding-bottom: 10px; }
        h2 { border-bottom: 1px solid #ddd; padding-bottom: 5px; margin-top: 30px; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        tr:nth-child(even) { background-color: #f9f9f9; }
        .metadata-table { width: 100%; }
        .metadata-table th { width: 30%; }
        .summary-table { width: 100%; }
        .summary-table th { width: 40%; }
        .chart-container { margin: 20px 0; }
        .map-container { height: 500px; margin: 20px 0; }
        
        @media print {
            body { font-size: 10pt; }
            .page-break { page-break-before: always; }
            table { page-break-inside: avoid; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <p>生成日時: {generated_date}</p>
        
        <h2>メタデータ</h2>
        {metadata_table}
        
        <h2>データサマリー</h2>
        {stats_table}
        
        <div class="page-break"></div>
        
        <h2>データテーブル</h2>
        <p>以下は記録されたデータの最初の100行です。</p>
        <div style="overflow-x: auto;">
            {data_table}
        </div>
        
        <h2>エクスポート情報</h2>
        <p>このレポートはセーリング戦略分析システムによって生成されました。</p>
        <p>生成日時: {generated_date}</p>
    </div>
</body>
</html>
"""