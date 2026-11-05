import "@fontsource-variable/inter";
import "@fontsource-variable/inter/wght-italic.css";
import "@fontsource-variable/jetbrains-mono";
import "./styles/fonts.css";
import "./styles/tokens.css";
import "./styles/app.css";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router";
import { App } from "./App";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { retry: 1, refetchOnWindowFocus: false },
    mutations: { retry: false },
  },
});

const container = document.getElementById("root");
if (container === null) {
  throw new Error("index.html has no #root element");
}

createRoot(container).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </QueryClientProvider>
  </StrictMode>,
);
