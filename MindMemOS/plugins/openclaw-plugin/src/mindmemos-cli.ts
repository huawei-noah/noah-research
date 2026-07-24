import { spawn } from "node:child_process";

type SpawnResult = {
  stdout: string;
  stderr: string;
};

export async function spawnFileJson<T>(command: string, args: string[]): Promise<T> {
  const result = await spawnFile(command, args);
  try {
    return JSON.parse(result.stdout) as T;
  } catch (error) {
    const firstLine = result.stdout.trim().split("\n")[0] ?? "";
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`failed to parse mindmemos JSON output: ${message}; stdout=${firstLine}`);
  }
}

export async function spawnFileOk(command: string, args: string[], stdin?: string): Promise<void> {
  await spawnFile(command, args, stdin);
}

function spawnFile(command: string, args: string[], stdin?: string): Promise<SpawnResult> {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      stdio: ["pipe", "pipe", "pipe"],
    });

    let stdout = "";
    let stderr = "";

    child.stdout.setEncoding("utf-8");
    child.stderr.setEncoding("utf-8");
    child.stdout.on("data", (chunk: string) => {
      stdout += chunk;
    });
    child.stderr.on("data", (chunk: string) => {
      stderr += chunk;
    });

    child.on("error", reject);
    child.on("close", (code) => {
      if (code === 0) {
        resolve({ stdout, stderr });
        return;
      }
      reject(new Error(`mindmemos exited with code ${code}: ${stderr || stdout}`.trim()));
    });

    if (stdin !== undefined) {
      child.stdin.end(stdin);
    } else {
      child.stdin.end();
    }
  });
}
