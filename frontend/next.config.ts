import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // The dev server treats 127.0.0.1 as cross-origin and blocks its own chunks; browser
  // tests and phone-on-the-LAN both hit that, so allow the loopback aliases explicitly.
  allowedDevOrigins: ["127.0.0.1", "localhost", "192.168.1.4"],
};

export default nextConfig;
