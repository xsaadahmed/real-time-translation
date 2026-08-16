/** @type {import('next').NextConfig} */
const nextConfig = {
  // Allow opening the dev UI via 127.0.0.1 (run_production.py prints that host).
  allowedDevOrigins: ['127.0.0.1', 'localhost'],
  typescript: {
    ignoreBuildErrors: true,
  },
  images: {
    unoptimized: true,
  },
}

export default nextConfig
