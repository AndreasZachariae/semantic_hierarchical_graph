import { AWS } from "./AWS";

export interface Path {
    [key: string]: AWS; // any string can be a key, a value is either an empty object or a Path
}