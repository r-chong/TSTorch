/* 
Modules form a tree that store parameters and other
submodules. They make up the basis of neural network stacks.

Attributes:
    modules : Storage of the child modules
    parameters : Storage of the module's parameters
    training : Whether the module is in training mode or evaluation mode
*/

export class Module {
    private modules: Record<string, Module> = {};
    private parameters: Record<string, BaseParameter> = {};
    training: boolean = true;

    // Automatically register submodules and parameters
    constructor() {
        return new Proxy(this, {
            set: (target, key: string, value) => {
                if (value instanceof Module) {
                    target.modules[key] = value;
                } 
                else if (value instanceof Parameter) {
                    target.parameters[key] = value;
                }
                else {
                    (target as any)[key] = value;
                }

                return true;
            }
        })
    }
}

// Non-generic base class to type Parameter class yet not Module class
export abstract class BaseParameter {
    name?: string;
}

// TODO: default T=Tensor when merging into Tensor
export class Parameter<T> extends BaseParameter {
    value: T;

    constructor(value: T, name?: string) {
        super();
        this.value = value;

        if (name) {
            this.name = name;
        }
    }

    update(v: T) {
        this.value = v;
    }
}
