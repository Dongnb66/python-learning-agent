import { handleLearningAgent } from "./handler.mjs";

Deno.serve((request: Request) => handleLearningAgent({ request, env: (n: string) => Deno.env.get(n) }));