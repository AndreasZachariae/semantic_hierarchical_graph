import { Position } from "./Position";

export interface AreaType {
    id: number;
    name: string;
    color: string;
}

/**
 * Represents an area on a floor.
 */
export interface Area {
    name: string;
    type: number;
    vertices: Position[];
}