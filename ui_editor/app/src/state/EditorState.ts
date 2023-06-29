import { create } from "zustand";

interface EditorState {
    zoom: number;
    setZoom: (zoom: number) => void;
    currentFloor: number;
    setCurrentFloor: (currentFloor: number) => void;
    editedElement: {
        type: "none" | "area" | "location",
        id: number,
    };
    setEditedElement: (editedElement: { type: "none" | "area" | "location", id: number }) => void;
}

export const useEditorState = create<EditorState>((set) => ({
    zoom: 1,
    setZoom: (zoom: number) => set({ zoom }),
    currentFloor: -1,
    setCurrentFloor: (currentFloor: number) => set({ currentFloor }),
    editedElement: {
        type: "none",
        id: -1,
    },
    setEditedElement: (editedElement: { type: "none" | "area" | "location", id: number }) => set({ editedElement }),
}));