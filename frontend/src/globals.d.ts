export {};

declare global {
  interface Window {
    WorkspaceUI: any;
    WorkspaceOptions: any;
    state: any;
    enter: (user: any) => Promise<void>;
    addEvent: (id: number, type: string, data: any) => void;
    renderRun: (run: any) => void;
  }
}
