/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  
  // 環境変数
  env: {
    NEXT_PUBLIC_APP_VERSION: process.env.NEXT_PUBLIC_APP_VERSION || '0.1.0',
    NEXT_PUBLIC_APP_NAME: process.env.NEXT_PUBLIC_APP_NAME || 'セーリング戦略分析システム',
  },
  
  // 画像最適化設定
  images: {
    domains: ['sailing-strategy-api.onrender.com'],
  },
  
  // API リクエストのリライト設定（開発環境用）
  async rewrites() {
    return process.env.NODE_ENV === 'development'
      ? [
          {
            source: '/api/:path*',
            destination: `${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/:path*`,
          },
        ]
      : [];
  },
  
  // 出力設定
  output: 'standalone',
}

module.exports = nextConfig