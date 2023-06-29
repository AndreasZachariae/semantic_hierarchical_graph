import { Room } from "./Room";

export interface AWS {
    [key: string]: {} | Room; // any string can be a key, a value is either an empty object or a Room
}