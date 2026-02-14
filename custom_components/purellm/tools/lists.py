"""Lists tool handler for PureLLM (shopping list, to-do list)."""
from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)


async def _get_items(hass: "HomeAssistant", entity_id: str, status: str) -> list[dict]:
    """Fetch items from a todo list by status."""
    result = await hass.services.async_call(
        "todo", "get_items",
        {"entity_id": entity_id, "status": status},
        blocking=True,
        return_response=True
    )
    if result and entity_id in result:
        return result[entity_id].get("items", [])
    return []


async def _remove_all_items(hass: "HomeAssistant", entity_id: str, status: str) -> int:
    """Remove all items with given status. Retries until empty. Returns count removed."""
    total_removed = 0
    for attempt in range(5):  # Max 5 passes to handle stragglers
        items = await _get_items(hass, entity_id, status)
        if not items:
            break
        for item in items:
            name = item.get("summary")
            if name:
                try:
                    await hass.services.async_call(
                        "todo", "remove_item",
                        {"entity_id": entity_id, "item": name},
                        blocking=True
                    )
                    total_removed += 1
                except Exception:
                    _LOGGER.debug("Failed to remove '%s', will retry", name)
    return total_removed


async def manage_list(
    arguments: dict[str, Any],
    hass: "HomeAssistant",
) -> dict[str, Any]:
    """Manage shopping list and to-do lists.

    Args:
        arguments: Tool arguments (action, item, list_name)
        hass: Home Assistant instance

    Returns:
        List operation result
    """
    action = arguments.get("action", "").lower()
    item = arguments.get("item", "").strip()
    list_name = arguments.get("list_name", "").lower().strip()

    try:
        # Get all todo entities
        all_states = hass.states.async_all()
        todo_lists = [s for s in all_states if s.entity_id.startswith("todo.")]

        # Find the right list
        target_list = None

        if list_name:
            # Match by name
            for todo in todo_lists:
                friendly = todo.attributes.get("friendly_name", "").lower()
                if list_name in friendly or list_name in todo.entity_id.lower():
                    target_list = todo.entity_id
                    break
        else:
            # Default to shopping list if exists
            for todo in todo_lists:
                if "shopping" in todo.entity_id.lower():
                    target_list = todo.entity_id
                    break
            # Otherwise use first todo list
            if not target_list and todo_lists:
                target_list = todo_lists[0].entity_id

        if not target_list and action != "list_all":
            return {"error": "No to-do lists found. Create a to-do list or shopping list in HA first."}

        if action == "add":
            if not item:
                return {"error": "Please specify an item to add"}

            await hass.services.async_call(
                "todo", "add_item",
                {"entity_id": target_list, "item": item},
                blocking=True
            )

            list_friendly = hass.states.get(target_list).attributes.get("friendly_name", "list")
            return {
                "success": True,
                "action": "added",
                "item": item,
                "list": list_friendly,
                "message": f"Added '{item}' to {list_friendly}"
            }

        elif action == "complete" or action == "check" or action == "done":
            if not item:
                return {"error": "Please specify an item to complete"}

            items = await _get_items(hass, target_list, "needs_action")

            # Find matching item (case insensitive partial match)
            item_lower = item.lower()
            matched_item = None
            for list_item in items:
                summary = list_item.get("summary", "").lower()
                if item_lower in summary or summary in item_lower:
                    matched_item = list_item.get("summary")
                    break

            if not matched_item:
                return {"error": f"Could not find '{item}' on the list"}

            await hass.services.async_call(
                "todo", "update_item",
                {"entity_id": target_list, "item": matched_item, "status": "completed"},
                blocking=True
            )

            list_friendly = hass.states.get(target_list).attributes.get("friendly_name", "list")
            return {
                "success": True,
                "action": "completed",
                "item": matched_item,
                "list": list_friendly,
                "message": f"Completed '{matched_item}' on {list_friendly}"
            }

        elif action == "remove" or action == "delete":
            if not item:
                return {"error": "Please specify an item to remove"}

            result = await hass.services.async_call(
                "todo", "get_items",
                {"entity_id": target_list},
                blocking=True,
                return_response=True
            )

            items = []
            if result and target_list in result:
                items = result[target_list].get("items", [])

            # Find matching item
            item_lower = item.lower()
            matched_item = None
            for list_item in items:
                summary = list_item.get("summary", "").lower()
                if item_lower in summary or summary in item_lower:
                    matched_item = list_item.get("summary")
                    break

            if not matched_item:
                return {"error": f"Could not find '{item}' on the list"}

            await hass.services.async_call(
                "todo", "remove_item",
                {"entity_id": target_list, "item": matched_item},
                blocking=True
            )

            list_friendly = hass.states.get(target_list).attributes.get("friendly_name", "list")
            return {
                "success": True,
                "action": "removed",
                "item": matched_item,
                "list": list_friendly,
                "message": f"Removed '{matched_item}' from {list_friendly}"
            }

        elif action == "remove_all" or action == "delete_all":
            if not item:
                return {"error": "Please specify an item to remove"}

            result = await hass.services.async_call(
                "todo", "get_items",
                {"entity_id": target_list},
                blocking=True,
                return_response=True
            )

            items = []
            if result and target_list in result:
                items = result[target_list].get("items", [])

            # Find ALL matching items
            item_lower = item.lower()
            matched_items = []
            for list_item in items:
                summary = list_item.get("summary", "").lower()
                if item_lower in summary or summary in item_lower:
                    matched_items.append(list_item.get("summary"))

            if not matched_items:
                return {"error": f"Could not find '{item}' on the list"}

            # Remove sequentially — HA todo backend doesn't handle parallel well
            for matched_item in matched_items:
                await hass.services.async_call(
                    "todo", "remove_item",
                    {"entity_id": target_list, "item": matched_item},
                    blocking=True
                )

            list_friendly = hass.states.get(target_list).attributes.get("friendly_name", "list")
            count = len(matched_items)
            return {
                "success": True,
                "action": "removed_all",
                "item": item,
                "count": count,
                "removed_items": matched_items,
                "list": list_friendly,
                "message": f"Removed {count} '{item}' item{'s' if count != 1 else ''} from {list_friendly}"
            }

        elif action == "show" or action == "get" or action == "read":
            # Check if user wants completed items
            show_completed = arguments.get("status", "").lower() in ("completed", "done", "checked")
            status_filter = "completed" if show_completed else "needs_action"

            items = await _get_items(hass, target_list, status_filter)

            list_friendly = hass.states.get(target_list).attributes.get("friendly_name", "list")
            status_label = "completed" if show_completed else "active"

            if not items:
                return {
                    "list": list_friendly,
                    "count": 0,
                    "items": [],
                    "message": f"{list_friendly} has no {status_label} items"
                }

            # Sort items alphabetically for display
            item_names = sorted([i.get("summary", "") for i in items], key=str.lower)
            return {
                "list": list_friendly,
                "count": len(items),
                "items": item_names,
                "status": status_label,
                "message": f"{list_friendly} has {len(items)} {status_label} item{'s' if len(items) != 1 else ''}: {', '.join(item_names)}"
            }

        elif action == "sort" or action == "alphabetize":
            # Check if user wants to sort completed items
            sort_completed = arguments.get("status", "").lower() in ("completed", "done", "checked")
            status_filter = "completed" if sort_completed else "needs_action"
            status_label = "completed" if sort_completed else "active"

            items = await _get_items(hass, target_list, status_filter)

            if len(items) < 2:
                return {"message": f"List has fewer than 2 {status_label} items, nothing to sort"}

            # Get item names and sort alphabetically
            item_names = [i.get("summary", "") for i in items]
            sorted_names = sorted(item_names, key=str.lower)

            # Check if already sorted
            if item_names == sorted_names:
                return {"message": f"{status_label.capitalize()} items are already sorted alphabetically"}

            # Remove all items (sequential + retry to guarantee empty)
            removed = await _remove_all_items(hass, target_list, status_filter)
            _LOGGER.debug("Sort: removed %d items before re-adding", removed)

            # Re-add in sorted order (sequential to preserve order)
            for name in sorted_names:
                await hass.services.async_call(
                    "todo", "add_item",
                    {"entity_id": target_list, "item": name},
                    blocking=True
                )
                if sort_completed:
                    await hass.services.async_call(
                        "todo", "update_item",
                        {"entity_id": target_list, "item": name, "status": "completed"},
                        blocking=True
                    )

            list_friendly = hass.states.get(target_list).attributes.get("friendly_name", "list")
            return {
                "success": True,
                "action": "sorted",
                "count": len(sorted_names),
                "list": list_friendly,
                "message": f"Alphabetized {len(sorted_names)} {status_label} items on {list_friendly}"
            }

        elif action == "clear":
            # Respect status parameter — clear active or completed items
            clear_completed = arguments.get("status", "").lower() in ("completed", "done", "checked")
            status_filter = "completed" if clear_completed else "needs_action"
            status_label = "completed" if clear_completed else "active"

            # Check if there are items to clear
            items = await _get_items(hass, target_list, status_filter)
            if not items:
                return {"message": f"No {status_label} items to clear"}

            # Remove all with retry loop to guarantee complete removal
            total_removed = await _remove_all_items(hass, target_list, status_filter)

            list_friendly = hass.states.get(target_list).attributes.get("friendly_name", "list")
            return {
                "success": True,
                "action": "cleared",
                "count": total_removed,
                "status": status_label,
                "list": list_friendly,
                "message": f"Cleared {total_removed} {status_label} item{'s' if total_removed != 1 else ''} from {list_friendly}"
            }

        elif action == "list_all":
            # Show all available lists
            lists_info = []
            for todo in todo_lists:
                friendly = todo.attributes.get("friendly_name", todo.entity_id)
                lists_info.append({"name": friendly, "entity_id": todo.entity_id})

            return {
                "count": len(lists_info),
                "lists": lists_info,
                "message": f"Found {len(lists_info)} list{'s' if len(lists_info) != 1 else ''}"
            }

        else:
            return {"error": f"Unknown list action: {action}. Use add, complete, remove, remove_all, show, or clear."}

    except Exception as err:
        _LOGGER.error("Error managing list: %s", err, exc_info=True)
        return {"error": f"List error: {str(err)}"}
