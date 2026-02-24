import { defineConfig } from "vite";
import { fileURLToPath } from "url";
import path from "path";

const tstorchSrc = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../tstorch/src"
);

export default defineConfig({
  resolve: {
    alias: [
      {
        find: /.*fast_ops\.js$/,
        replacement: fileURLToPath(
          new URL("./src/stubs/fast_ops.ts", import.meta.url)
        ),
      },
      {
        find: "tstorch",
        replacement: path.join(tstorchSrc, "index.ts"),
      },
    ],
  },
});
