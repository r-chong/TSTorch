import { readFile } from 'node:fs/promises';
import * as ts from 'typescript';

export async function resolve(specifier, context, defaultResolve) {
    const isRelative =
        specifier.startsWith('.') || specifier.startsWith('/');

    if (isRelative && specifier.endsWith('.js')) {
        const tsSpecifier = specifier.slice(0, -3) + '.ts';
        try {
            return await defaultResolve(tsSpecifier, context, defaultResolve);
        } catch {
            // Fall back to default resolution for .js
        }
    }

    return defaultResolve(specifier, context, defaultResolve);
}

export async function load(url, context, defaultLoad) {
    if (url.endsWith('.ts')) {
        const source = await readFile(new URL(url), 'utf8');
        const transpiled = ts.transpileModule(source, {
            compilerOptions: {
                module: ts.ModuleKind.ESNext,
                target: ts.ScriptTarget.ES2022,
            },
        });

        return {
            format: 'module',
            source: transpiled.outputText,
            shortCircuit: true,
        };
    }

    return defaultLoad(url, context, defaultLoad);
}
