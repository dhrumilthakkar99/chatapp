import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import { ClerkProvider } from "@clerk/clerk-react";
import App from "./App";
import "./index.css";

const clerkPublishableKey = import.meta.env.VITE_CLERK_PUBLISHABLE_KEY;

if (!clerkPublishableKey) {
  ReactDOM.createRoot(document.getElementById("root")!).render(
    <React.StrictMode>
      <main style={{ padding: "2rem", color: "#e2e8f0", fontFamily: "sans-serif", background: "#0f1115", minHeight: "100vh" }}>
        Missing <code>VITE_CLERK_PUBLISHABLE_KEY</code>. Add it to <code>apps/web/.env.local</code>.
      </main>
    </React.StrictMode>
  );
} else {
  ReactDOM.createRoot(document.getElementById("root")!).render(
    <React.StrictMode>
      <ClerkProvider publishableKey={clerkPublishableKey} afterSignOutUrl="/">
        <BrowserRouter>
          <App />
        </BrowserRouter>
      </ClerkProvider>
    </React.StrictMode>
  );
}
