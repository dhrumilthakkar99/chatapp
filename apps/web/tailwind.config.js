/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "#0b1017",
        panel: "#121a24",
        panel2: "#1a2330",
        border: "#2a3645",
        accent: "#59d0f8",
      },
      boxShadow: {
        panel: "0 8px 24px rgba(0,0,0,0.25)",
      },
    },
  },
  plugins: [],
};
