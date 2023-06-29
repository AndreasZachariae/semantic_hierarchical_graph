import React from "react";
import { Position } from "./Position";

/**
 * Identifies what a location is used for.
 */
export interface LocationType {
    id: number;
    name: string;
    color: string;
    icon: () => React.ReactNode;
}

/**
 * Represents a location on a floor.
 */
export interface Location {
    name: string;
    position: Position;
    type: number;
}