from typing import Final
from enum import StrEnum
from dct import benchmark, plot
from app import DCT2App

class Command(StrEnum):
  """
  Available command.
  """
  INFO = "info"
  """
  Displays the identifier string of the engine.
  """
  HELP = "help"
  """
  | Displays the list of available commands.
  | If a command is specified, displays the help for that command.
  """
  DCT = "dct"
  """
  Compares a naive implementation of the DCT2 to SciPy's implementation.
  """
  BMP = "bmp"
  """
  Launches the application window to select and compress a .bmp image with JPEG compression type.
  """
  EXIT = "exit"
  """
  Exits the engine.
  """

class Engine:
  """
  Program engine.
  """

  VERSION: Final[str] = "0.0.1"
  """
  Engine version.
  """

  def start(self) -> None:
    """
    Engine main loop to handle commands.
    """
    self.info()
    while True:
      match input().strip().split():
        case [Command.INFO]:
          self.info()
        case [Command.HELP, *arguments]:
          self.help(arguments)
        case [Command.DCT, *arguments]:
          self.dct(arguments)
        case [Command.BMP]:
          self.bmp()
        case [Command.EXIT]:
          break
        case _:
          self.error("Invalid command. Try 'help' to see a list of valid commands and how to use them.")

  def info(self) -> None:
    """
    Handles 'info' command.
    """
    print(f"EngineDCT2 v{Engine.VERSION}")

  def help(self, arguments: list[str]) -> None:
    """
    Handles 'help' command with arguments.

    :param arguments: Command arguments.
    :type arguments: list[str]
    """
    if arguments:
      if len(arguments) > 1:
        self.error(f"Too many arguments for command '{Command.HELP}'")
      else:
        match arguments[0]:
          case Command.INFO:
            print(f"  {Command.INFO}")
            print()
            print("  Displays the identifier string of the engine.")
          case Command.HELP:
            print(f"  {Command.HELP}")
            print(f"  {Command.HELP} [command]")
            print()
            print("  Displays the list of available commands. If a command is specified, displays the help for that command.")
          case Command.DCT:
            print(f"  {Command.DCT}")
            print(f"  {Command.DCT} [n]")
            print()
            print("  Compares a naive implementation of the DCT2 to SciPy's implementation. The comparison runs for N×N arrays with N starting from 2³ and doubling up to 2ⁿ (n defaults to 12).")
          case Command.BMP:
            print(f"  {Command.BMP}")
            print()
            print("  Launches the application window to select and compress a .bmp image with JPEG compression type.")
          case Command.EXIT:
            print(f"  {Command.EXIT}")
            print("")
            print("  Exits the engine.")
          case _:
            self.error(f"Unknown command '{arguments[0]}'")
    else:
      print("Available commands:")
      for command in Command:
        print(f"  {command}")
      print(f"Try '{Command.HELP} <command>' to see help for a particular Command")

  def dct(self, arguments: list[str]) -> None:
    """
    Handles the 'dct' command with arguments.

    :param arguments: Command arguments.
    :type arguments: list[str]
    """
    if not arguments or len(arguments) == 1:
      result = benchmark([2**i for i in range(3, (max(3, int(arguments[0])) if arguments else 12) + 1)])
      print("Benchmark summary:\n", result)
      plot(result).show()
      print()
    else:
      self.error(f"Too many arguments for command '{Command.HELP}'")

  def bmp(self) -> None:
    """
    Handles the 'bmp' command.
    """
    DCT2App().mainloop()

  def error(self, error: str | Exception) -> None:
    """
    Outputs as an error the given message/exception.

    :param message: Message or exception.
    :type message: str | Exception
    """
    print(f"err {error}.")

if __name__ == "__main__":
  Engine().start()
