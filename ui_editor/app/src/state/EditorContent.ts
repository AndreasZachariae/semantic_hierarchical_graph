import { create } from "zustand";
import { Floor } from "../types/Floor";

interface FloorContainer {
    floors: Floor[];
    setFloors: (floors: Floor[]) => void;
    addFloor: (floor: Floor) => Promise<Floor>;
    setFloor: (floor: Floor) => void;
    removeFloor: (floor: Floor) => void;
}

 export const useFloors = create<FloorContainer>((set) => ({
    floors: [],
    setFloors: (floors: Floor[]) => set({ floors }),
    addFloor: (floor: Floor) => {
        return new Promise((resolve) => set((state: FloorContainer) => {
            floor.id = state.floors.map((f) => f.id || 0).reduce((a, b) => Math.max(a, b), 0) + 1;
            resolve(floor);
            return { floors: [...state.floors, floor] }
        }));
    },
    setFloor: (floor: Floor) => set((state: FloorContainer) => {
            const index = state.floors.findIndex((f) => f.id === floor.id);
            const floors = [...state.floors];
            floors[index] = floor;
            return { floors };
        }),
    removeFloor: (floor: Floor) => set((state: FloorContainer) => ({ floors: state.floors.filter((f) => f.id !== floor.id) })),
}));