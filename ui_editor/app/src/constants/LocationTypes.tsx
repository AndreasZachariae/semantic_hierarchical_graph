import { HomeWork, Hotel, MeetingRoom, Power, Warehouse, Wc } from "@mui/icons-material";
import { LocationType } from "../types/Location";

export const Room: LocationType = {
    id: 0,
    name: "Room",
    color: "#ff0000",
    icon: () => <MeetingRoom />,
}

export const Bed: LocationType = {
    id: 1,
    name: "Bed",
    color: "#00ff00",
    icon: () => <Hotel />,
}

export const Office: LocationType = {
    id: 2,
    name: "Office",
    color: "#0000ff",
    icon: () => <HomeWork />,
}

export const Toilet: LocationType = {
    id: 3,
    name: "Toilet",
    color: "#ffff00",
    icon: () => <Wc />,
}

export const Storage: LocationType = {
    id: 4,
    name: "Storage",
    color: "#ff00ff",
    icon: () => <Warehouse />,
}

export const Charger: LocationType = {
    id: 5,
    name: "Charger",
    color: "#00ffff",
    icon: () => <Power />,
}

export const LocationTypes: LocationType[] = [
    Room,
    Bed,
    Office,
    Toilet,
    Storage,
    Charger,
];

export function getLocationTypeById(id: number): LocationType {
    return LocationTypes.filter(type => type.id === id)[0];
}