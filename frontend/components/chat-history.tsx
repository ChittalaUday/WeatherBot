"use client";

import { useQuery } from "@tanstack/react-query";
import { History, MessageSquare } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import { apiUrl, cn } from "@/lib/utils";


type ChatSummary = {
  chat_id: string;
  title: string;
  turns: number;
  answered: number;
  last_active: string;
  model: string;
};

const ago = (iso: string) => {
  // the store writes UTC with an explicit offset; only bare timestamps need the Z
  const stamp = /[Z+]|-\d{2}:\d{2}$/.test(iso) ? iso : `${iso}Z`;
  const parsed = new Date(stamp).getTime();
  if (Number.isNaN(parsed)) return "";
  const minutes = Math.max(0, Math.round((Date.now() - parsed) / 60000));
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.round(minutes / 60);
  return hours < 24 ? `${hours}h ago` : `${Math.round(hours / 24)}d ago`;
};

/** Past conversations. Opening one replays the answers as they were rendered, rather than
 *  re-querying a forecast that has since moved on. */
export function ChatHistory({
  currentChatId,
  onOpenChat,
}: {
  currentChatId: string;
  onOpenChat: (chatId: string) => void;
}) {
  const { data, isFetching, refetch } = useQuery({
    queryKey: ["chats"],
    queryFn: async () => (await fetch(`${apiUrl()}/api/chats?limit=40`)).json(),
    staleTime: 15_000,
  });
  const chats: ChatSummary[] = data?.chats ?? [];

  return (
    <Sheet onOpenChange={(open) => open && refetch()}>
      {/* this shadcn build is on Base UI, where the trigger takes `render`, not `asChild` */}
      <SheetTrigger
        render={
          <Button
            variant="ghost"
            size="sm"
            title="Past chats"
            className="h-8 gap-1.5 px-2 text-xs text-muted-foreground"
          >
            <History className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">History</span>
          </Button>
        }
      />
      <SheetContent side="left" className="w-[330px] p-0 sm:w-[380px]">
        <SheetHeader className="px-4 pb-2 pt-4">
          <SheetTitle className="text-base">Chats</SheetTitle>
          <SheetDescription className="text-xs">
            {isFetching ? "loading…" : `${chats.length} conversation${chats.length === 1 ? "" : "s"}`}
          </SheetDescription>
        </SheetHeader>
        <ScrollArea className="h-[calc(100dvh-84px)] px-2 pb-4">
          <div className="flex flex-col gap-1">
            {chats.map((chat) => (
              <button
                key={chat.chat_id}
                onClick={() => onOpenChat(chat.chat_id)}
                className={cn(
                  "flex w-full flex-col gap-1 rounded-xl px-3 py-2.5 text-left transition-colors hover:bg-muted",
                  chat.chat_id === currentChatId && "bg-muted",
                )}
              >
                <span className="flex items-center gap-1.5">
                  <MessageSquare className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
                  <span className="truncate text-sm">{chat.title}</span>
                </span>
                <span className="pl-5 text-[11px] text-muted-foreground">
                  {chat.turns} turn{chat.turns === 1 ? "" : "s"} · {chat.model} · {ago(chat.last_active)}
                </span>
              </button>
            ))}
            {chats.length === 0 && !isFetching && (
              <p className="px-3 py-6 text-center text-xs text-muted-foreground">
                No chats yet. Ask something and it will appear here.
              </p>
            )}
          </div>
        </ScrollArea>
      </SheetContent>
    </Sheet>
  );
}
