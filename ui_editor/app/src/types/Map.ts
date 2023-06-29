import { Position } from "./Position";

/**
 * A map is used to represent the scanned area of a floor.
 */
export interface Map {
    file: string;
    resolution: number;
    width: number;
    height: number;
    origin: Position;
}