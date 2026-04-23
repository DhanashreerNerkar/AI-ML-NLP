import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { Analytics } from "@vercel/analytics/next";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Vectorless PDF Chatbot",
  description: "Chat with your PDF without RAG",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        <div className="min-h-screen flex flex-col">
          <main className="flex-1 bg-gray-50">
            {children}
          </main>
          <footer className="bg-gray-100 border-t border-gray-200 py-6">
            <div className="container mx-auto px-4 text-center">
              <p className="text-gray-600 text-sm">
                Made with{" "}
                <span className="text-red-500 text-base">♥</span>{" "}
                by{" "}
                <span className="font-semibold text-gray-800">ROE AI Inc.</span>
              </p>
            </div>
          </footer>
        </div>
        <Analytics />
      </body>
    </html>
  );
}
