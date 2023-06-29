import { Area } from "./Area";
import { Location } from "./Location";
import { Map } from "./Map";

/**
 * A floor is used to represent a floor of a building.
 * It contains a map aswell as all objects on the floor.
 */
export interface Floor {
    id?: number;
    name: string;
    map: Map;
    areas: Area[];
    locations: Location[];
}